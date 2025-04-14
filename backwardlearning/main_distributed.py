import os
import json
import logging
import argparse
from typing import List
import datetime
import jsonlines

import wandb
import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
from vllm import SamplingParams
from transformers import AutoTokenizer

from backwardlearning.utils import (
    launch_fastchat_server,
    launch_fastchat_server_worker,
    launch_fastchat_tp,
    save_states,
    stop_fastchat_server,
    log_execution_time,
    rename_file,
    State,
)
from backwardlearning.learner import Learner
from backwardlearning.dpo_runner import DPORunner
from backwardlearning.dist_utils import file_system_barrier, file_system_broadcast

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s, line %(lineno)d in %(funcName)s %(filename)s - %(message)s',
)
logger = logging.getLogger(__file__)

TRAINER_STATE_NAME = "trainer_state.json"
TRAINER_STATE_LOG_NAME = "trainer_state_log.json"


def run_batch(
    args,
    learner: Learner,
    states_batch: List[List[State]],
    dpo_runner: DPORunner,
    is_main_process: bool,
    world_size: int,
    rank: int,
    master_addr: str,
    dpo_master_port: int,
    worker_addr: str,
    cycle_id: int,
    batch_id: int,
    batch_epoch_idx: int,
    tot_batch_num: int,
    plot: bool=True,
    evaluate: bool=True,
    eval_interval: int=10,
):

    dataset_dir = f"{args.train_dataset_dir}/{args.exp_name}"
    dataset_name = f"{dataset_dir}/hf-cycle_{cycle_id}_{batch_id}_{batch_epoch_idx}.json"
    if not args.restart_train_cycle:
        model_path = args.model_path if cycle_id == 0 and batch_id == 0 and batch_epoch_idx == 0 else f"{args.train_ckpt_dir}/{args.exp_name}/{tot_batch_num-1}"
        output_path = f"{args.train_ckpt_dir}/{args.exp_name}/{tot_batch_num}"
    else:
        model_path = args.model_path if cycle_id == 0 and batch_id == 0 and batch_epoch_idx == 0 else f"{args.train_ckpt_dir}/{args.exp_name}/{tot_batch_num-1}"
        output_path = f"{args.train_ckpt_dir}/{args.exp_name}/{tot_batch_num}"
    if args.dpo_log_kl:
        metric_log_file = f"{output_path}/metrics.json"
    else:
        metric_log_file = None

    if is_main_process or args.logic_all_nodes:
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)
        all_states = sum(states_batch, [])
        with log_execution_time(f" cycle {cycle_id} batch {batch_id} epoch {batch_epoch_idx}: evaluation"):
            if args.use_first:
                learner.evaluate_states([query_states[0] for query_states in states_batch])
            else:
                learner.evaluate_states([state for state in all_states if state.q < args.evaluation_max_p])
        buffer_states = []
        all_qs_train = []
        for query_states in states_batch:
            if args.use_first:
                buffer_states.append([query_states[0]])
            else:
                buffer_states.append([state for state in query_states if state.q < learner.buffer_max_p and state.q > learner.buffer_min_p])
            all_qs_train.append([state.q for state in query_states])
        os.makedirs(f"figs/{args.exp_name}", exist_ok=True)
        np.savez(f"figs/{args.exp_name}/train_{cycle_id}_{batch_id}_{batch_epoch_idx}.npz", *all_qs_train)
        if plot:
            plt.figure(figsize=(12, 12))
            for traj_idx, qs in enumerate(all_qs_train):
                plt.plot(np.arange(len(qs)) / len(qs), qs, label=traj_idx)
            plt.legend()
            os.makedirs(f"figs/{args.exp_name}", exist_ok=True)
            plt.savefig(f"figs/{args.exp_name}/train_{cycle_id}_{batch_id}_{batch_epoch_idx}.png")
        if args.use_wandb:
            wandb.log({
                "train/q0": np.mean([qs[0] for qs in all_qs_train]),
                "train/qmean": np.mean([np.mean(qs) for qs in all_qs_train]),
            }, step=tot_batch_num)

        if args.dpo_dataset_strategy == "closest_0.5":
            closest_states = []
            for states, qs in zip(buffer_states, all_qs_train):
                closest_idx = np.argmin(np.abs(np.array(qs) - 0.5))
                closest_states.append(states[closest_idx])
            learner.gen_dpo_dataset(dataset_name, closest_states, args.dpo_dataset_strategy)
        elif args.dpo_dataset_strategy == "vine_reinforce":
            all_query_ids = [states[0].query_id for states in states_batch]
            learner.gen_vine_dataset(dataset_name, all_query_ids)
        else:
            learner.gen_dpo_dataset(dataset_name, sum(buffer_states, []),
                                    pairing_method=args.dpo_dataset_strategy,
                                    filter_by_length=args.filter_trajs_by_length,
                                    filter_by_ans_ridx=args.filter_trajs_by_ansidx)
        save_states(all_states, f"{dataset_dir}/all_states_{tot_batch_num}.dataset")

        # if args.update_trajs and (tot_batch_num + 1) % args.update_trajs_interval == 0:
        #     learner.update_queries_traj(logfile=f"{dataset_dir}/replaced_trajs_{cycle_id}_{batch_id}_{batch_epoch_idx}.json", states=states_batch)

    else:
        pass

    resume_from_checkpoint = (not args.restart_train_cycle) and (tot_batch_num > 0)
    if is_main_process and resume_from_checkpoint:
        rename_file(model_path, TRAINER_STATE_NAME, TRAINER_STATE_LOG_NAME)
    if args.distribute_type == "torch":
        dist.barrier()
    elif args.distribute_type == "file":
        file_system_barrier(args.shared_file_path, rank, world_size, f"training_barrier-{tot_batch_num}")
    stop_fastchat_server()
    # do dpo training
    with log_execution_time(f"cycle {cycle_id} batch {batch_id} batch epoch {batch_epoch_idx} rank {rank}: dpo training"):
        dpo_runner.run_dpo_distributed(
            dataset=dataset_name,
            model_path=model_path,
            output_dir=output_path,
            num_train_epochs=args.dpo_train_epochs,
            world_size=world_size,
            master_addr=master_addr,
            main_process_port=dpo_master_port,
            rank=rank,
            batch_size=args.dpo_batch_size,
            beta=args.dpo_beta,
            per_device_train_batch_size=1,
            rpo_alpha=args.dpo_rpo_alpha,
            ref_model=args.model_path if args.dpo_fix_ref_model else None,
            metric_log_file=metric_log_file,
            resume_model_path=model_path if resume_from_checkpoint else None,
            learning_rate=args.dpo_lr,
            lr_scheduler_type=args.dpo_lr_scheduler_type,
            torch_dtype=args.dpo_torch_dtype,
            max_length=args.dpo_max_length,
            max_prompt_length=args.dpo_max_prompt_length,
            method=args.dpo_method,
            loss_type=args.dpo_loss_type,
            minibatch_accum=args.evaluation_trajs_num // 2 if args.dpo_method in ["grreinforce", "grpo"] else 1,
            no_shuffle=args.dpo_no_shuffle
        )

    # launch vllm server
    if is_main_process:
        # launch_fastchat_server()
        launch_fastchat_tp(
            output_path if not args.fix_generation else args.model_path,
            master_addr,
            worker_addr,
            is_main_process,
            num_gpus=torch.cuda.device_count(),
            session_name="llm_server",
            wait_secs=args.vllm_wait_sec,
            log_dir=args.vllm_logdir,
            vllm_tp=args.vllm_tp
        )
        logger.info("launched fastchat server cycle id %d batch id %d epoch %d from master", cycle_id, batch_id, batch_epoch_idx)

        if ((1 + tot_batch_num) % eval_interval == 0) and evaluate:
            # evaluate all eval state
            all_states = []
            for key, state_pool in learner.eval_query_state_pool.items():
                for state in state_pool.states:
                    all_states.append(state)
            learner.evaluate_states(all_states, custom_sampling_params=learner.eval_sampling_params)
            all_qs = []
            all_q0, all_qmean = [], []
            for traj_idx, key in enumerate(learner.eval_query_state_pool.keys()):
                qs = np.array([state.q for state in learner.eval_query_state_pool[key].states])
                plt.plot(np.arange(len(qs)) / len(qs), qs, label=traj_idx)
                all_qs.append(qs)
                all_q0.append(qs[0])
                all_qmean.append(qs.mean())
            if plot:
                plt.figure(figsize=(12, 12))
                plt.legend()
                os.makedirs(f"figs/{args.exp_name}", exist_ok=True)
                plt.savefig(f"figs/{args.exp_name}/eval_{cycle_id}_{batch_id}_{batch_epoch_idx+1}.png")
                np.savez(f"figs/{args.exp_name}/eval_{cycle_id}_{batch_id}_{batch_epoch_idx+1}.npz", *all_qs)
            if args.use_wandb:
                wandb.log({
                    "eval/q0": np.mean(all_q0),
                    "eval/qmean": np.mean(all_qmean)
                }, step=tot_batch_num+1)
            save_states(all_states, f"{dataset_dir}/eval_states_{tot_batch_num}.dataset")

        if args.dpo_log_kl:
            with open(metric_log_file, "r") as f:
                dporun_metric = json.load(f)
            kl_raw_est = np.array(dporun_metric['train']['rewards/rejected'])
            r = np.exp(kl_raw_est)
            kl_exp_est = (r - 1) - kl_raw_est
            wandb.log({
                "kl/raw_mean": kl_raw_est.mean(),
                "kl/raw_last": kl_raw_est[-1],
                "kl/exp_mean": kl_exp_est.mean(),
                "kl/exp_last": kl_exp_est[-1],
            }, step=tot_batch_num+1)

        os.system(f"rm -rf {model_path}/checkpoint-*")
    else:
        launch_fastchat_tp(
            output_path if not args.fix_generation else args.model_path,
            master_addr,
            worker_addr,
            is_main_process,
            num_gpus=torch.cuda.device_count(),
            session_name="llm_server",
            wait_secs=3,
            log_dir=args.vllm_logdir,
            vllm_tp=args.vllm_tp
        )
        logger.info("launched fastchat server cycle id %d batch id %d epoch %d, from rank %d", cycle_id, batch_id, batch_epoch_idx, rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logic_all_nodes", action="store_true", default=False, help="run state selection, dataset generation on all states")
    parser.add_argument("--model_path", type=str, default="/cpfs_29g2ues/data/shared/Group-m6/gaojingyue.gjy/qstar_reasoning_ckpt/7b.exp7_Math_500B_tokens_v2_avg9best_math_aops_v1_bsz128_epoch3-7B.qwen2B-bf16-mp4-pp1-lr-3e-5-minlr-7e-7-bs-128-gpus-256-seqlen-4096-step3960")
    parser.add_argument("--model_type", choices=["qwen", "llama3", "mistral", "metamath", "qwenmath", "acemath"], default="qwen")
    parser.add_argument("--datafile_path", type=str, default="data/trajectories/math_correct_32.json")
    parser.add_argument("--eval_datafile_path", type=str, default="data/trajectories/math_correct_32.json")
    parser.add_argument("--model_name", type=str, default="qwen2-7b-math")
    parser.add_argument("--distribute_type", choices=["file", "torch"], default="torch")
    parser.add_argument("--step_based_states", action="store_true", default=False)
    parser.add_argument("--step_states_skip", type=int, default=1)
    parser.add_argument("--states_number", type=int, default=5)
    parser.add_argument("--use_first", action="store_true", default=False)
    parser.add_argument("--fix_generation", action="store_true", default=False)

    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=20)
    parser.add_argument("--plot_first", action="store_true", default=False)
    parser.add_argument("--plot_fig", action="store_true", default=False)
    parser.add_argument("--regenerate_trajs", action="store_true", default=False)
    parser.add_argument("--dataset_min_p", type=float, default=0.3)
    parser.add_argument("--dataset_max_p", type=float, default=0.7)
    parser.add_argument("--evaluation_max_p", type=float, default=0.99)
    parser.add_argument("--evaluation_trajs_num", type=int, default=64)

    parser.add_argument("--train_script", type=str, default="backwardlearning/trl_dpo_nll.py")
    parser.add_argument("--train_dataset_dir", type=str, default="data/dpo_noheap")
    parser.add_argument("--train_dataset_size", type=int, default=-1)
    parser.add_argument("--train_ckpt_dir", type=str, default="/cpfs_29g2ues/data/shared/Group-m6/gaojingyue.gjy/qstar_reasoning_ckpt/dpo_noheap")

    parser.add_argument("--restart_train_cycle", action="store_true", default=False)
    parser.add_argument("--dpo_train_epochs", type=int, default=1)
    parser.add_argument("--dpo_batch_size", type=int, default=128)
    parser.add_argument("--dpo_master_port", type=int, default=6789)
    parser.add_argument("--dpo_rpo_alpha", type=float, default=None)
    parser.add_argument("--dpo_beta", type=float, default=0.05)
    parser.add_argument("--dpo_lr", type=float, default=5e-7)
    parser.add_argument("--dpo_lr_scheduler_type", choices=['constant', 'linear', 'cosine'], default='cosine')
    parser.add_argument("--dpo_dataset_strategy", choices=["short", "single_sample", "closest_0.5", "grreinforce", "vine_reinforce"], default="single_sample")
    parser.add_argument("--dpo_log_kl", action="store_true", default=False)
    parser.add_argument("--dpo_fix_ref_model", action="store_true", default=False)
    parser.add_argument("--dpo_offload", action="store_true", default=False)
    parser.add_argument("--dpo_accelerate_config_file", type=str, default="configs/accelerate_configs/deepspeed_zero3.yaml")
    parser.add_argument("--dpo_torch_dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--dpo_max_length", type=int, default=2048)
    parser.add_argument("--dpo_loss_type", type=str, default="sigmoid")
    parser.add_argument("--dpo_method", type=str, default="qstar")
    parser.add_argument("--dpo_max_prompt_length", type=int, default=1024)
    parser.add_argument("--dpo_no_shuffle", action="store_true", default=False)

    parser.add_argument("--query_batch_num_max", type=int, default=-1)
    parser.add_argument("--query_batch_size", type=int, default=64)
    parser.add_argument("--query_train_epoch", type=int, default=1)
    parser.add_argument("--query_resume_batch", type=int, default=0)
    parser.add_argument("--query_resume_epoch", type=int, default=0)
    parser.add_argument("--query_batch_mode", choices=["iteration", "queue"], default="iteration")
    parser.add_argument("--query_timeout_iter", type=int, default=15)
    parser.add_argument("--eval_interval", type=int, default=10)

    parser.add_argument("--vllm_logdir", type=str, default="log")
    parser.add_argument("--vllm_post_concurrency", type=int, default=64)
    parser.add_argument("--vllm_tp", type=int, default=1)
    parser.add_argument("--vllm_wait_sec", type=int, default=100)

    parser.add_argument("--init_trajs_regenerate", action="store_true", default=False)
    parser.add_argument("--init_trajs_number", type=int, default=16)
    parser.add_argument("--update_trajs", action="store_true", default=False)
    parser.add_argument("--update_trajs_interval", type=int, default=1)
    parser.add_argument("--update_trajs_method", choices=["random", "advantage", "max_prob", "preference"], default="random")
    parser.add_argument("--filter_trajs_by_length", action="store_true", default=False)
    parser.add_argument("--filter_trajs_by_ansidx", action="store_true", default=False)
    parser.add_argument("--flaming_decoding", action="store_true", default=False)

    parser.add_argument("--exp_name", type=str, default="dpo")
    parser.add_argument("--exp_name_appendtime", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--shared_file_path", type=str, default=".distributed")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    args = parser.parse_args()

    if args.logic_all_nodes:
        raise NotImplementedError("logic on all nodes parallelly not implemented yet")

    if args.exp_name_appendtime:
        now = datetime.now()
        strftime = now.strftime("%Y%m%d-%H%M%S")
        args.exp_name = f"{args.exp_name}_{strftime}"

    master_addr = os.environ["MASTER_ADDR"]
    master_port = int(os.environ["MASTER_PORT"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.distribute_type == "torch":
        dist.init_process_group(
            backend="gloo",
            # init_method=f"tcp://{master_addr}:10010",
            world_size=world_size,
            rank=int(os.environ["RANK"]),
            timeout=datetime.timedelta(hours=5)
        )
        print(dist.is_initialized())
    rank = int(os.environ["RANK"])
    is_main_process = rank == 0
    if args.distribute_type == "torch":
        worker_addr = f"{os.environ['JOB_NAME']}-worker-{int(os.environ['RANK'])-1}" if not is_main_process else None
    elif args.distribute_type == "file":
        worker_addr = f"{os.environ['KUBERNETES_POD_NAME']}" if not is_main_process else None
    else:
        worker_addr = None
    args.use_wandb = args.use_wandb and is_main_process

    if args.use_wandb:
        wandb.init(
            config=vars(args),
            name=args.exp_name,
            project=args.wandb_project,
            reinit=True,
            entity=args.wandb_entity
        )

    if is_main_process:
        data_file = args.datafile_path
        with jsonlines.open(data_file, "r") as f:
            all_trajs = list(f)

        trajs = all_trajs
        quries = [accurate_traj['query'] for accurate_traj in trajs]
        if not args.init_trajs_regenerate:
            responses = [accurate_traj[''] for accurate_traj in trajs]
        gts = [accurate_traj['gt'] for accurate_traj in trajs]

        if args.eval:
            eval_data_file = args.eval_datafile_path
            with jsonlines.open(eval_data_file, "r") as f:
                eval_trajs = list(f)

            eval_queries = [accurate_traj['query'] for accurate_traj in eval_trajs]
            eval_responses = [accurate_traj[''] for accurate_traj in eval_trajs]
            eval_gts = [accurate_traj['gt'] for accurate_traj in eval_trajs]

    if args.logic_all_nodes:
        # todo: spreading datafile across nodes
        pass

    if args.distribute_type == "torch":
        dist.barrier()
    elif args.distribute_type == "file":
        file_system_barrier(args.shared_file_path, rank, world_size, "launch fastchat")

    if is_main_process:
        launch_fastchat_tp(
            args.model_path,
            master_addr,
            worker_addr,
            is_main_process,
            num_gpus=torch.cuda.device_count(),
            session_name="llm_server",
            wait_secs=args.vllm_wait_sec,
            log_dir=args.vllm_logdir,
            vllm_tp=args.vllm_tp
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        learner = Learner(
            args=args,
            queries=quries,
            gts=gts,
            model_name=args.model_name,
            model_path=args.model_path,
            dpo_script_name=args.train_script,
            dpo_dataset_dir=args.train_dataset_dir,
            dpo_output_dir=args.train_ckpt_dir,
            buffer_max_p=args.dataset_max_p,
            buffer_min_p=args.dataset_min_p,
            vllm_server_url="http://localhost:8000/v1/",
            vllm_server_num_gpus=torch.cuda.device_count(),
            vllm_server_post_num_threads=args.vllm_post_concurrency,
            vllm_log_dir=args.vllm_logdir,
            init_trajs=None if args.init_trajs_regenerate else responses,
            init_trajs_only_correct=args.update_trajs_method != "max_prob",
            query_batch_size=args.query_batch_size,
            query_batch_mode=args.query_batch_mode,
            evaluation_trajs_num=args.evaluation_trajs_num,
            eval_dataset=(eval_queries, eval_gts, eval_responses) if args.eval else None,
            query_timeout_iter=args.query_timeout_iter,
            use_step_based=args.step_based_states,
            vine_reinforce=args.dpo_method == "vine_reinforce",
            tokenizer=tokenizer
        )
        learner.save_states_pool(f"{args.train_dataset_dir}/{args.exp_name}/inital_state_pool.pkl")
        dpo_runner = DPORunner(
            args.train_script,
            config_file=args.dpo_accelerate_config_file,
            num_process=torch.cuda.device_count(),
        )
    else:
        learner = None
        launch_fastchat_tp(
            args.model_path,
            master_addr,
            worker_addr,
            is_main_process,
            num_gpus=torch.cuda.device_count(),
            session_name="llm_server",
            wait_secs=1,
            log_dir=args.vllm_logdir,
            vllm_tp=args.vllm_tp
        )
        dpo_runner = DPORunner(
            args.train_script,
            config_file=args.dpo_accelerate_config_file,
            num_process=torch.cuda.device_count(),
        )

    if args.distribute_type == "torch":
        dist.barrier()
    elif args.distribute_type == "file":
        file_system_barrier(args.shared_file_path, rank, world_size, "wait_for_all_nodes")

    if is_main_process and args.eval:
        # evaluate all eval state
        all_states = []
        for key, state_pool in learner.eval_query_state_pool.items():
            for state in state_pool.states:
                all_states.append(state)
        print(len(all_states))

        learner.evaluate_states(all_states, custom_sampling_params=learner.eval_sampling_params)
        if args.use_wandb:
            all_qs = []
            all_q0, all_qmean = [], []
            for traj_idx, key in enumerate(learner.eval_query_state_pool.keys()):
                qs = np.array([state.q for state in learner.eval_query_state_pool[key].states])
                all_qs.append(qs)
                all_q0.append(qs[0])
                all_qmean.append(qs.mean())
            wandb.log({
                "eval/q0": np.mean(all_q0),
                "eval/qmean": np.mean(all_qmean)
            }, step=0)
            os.makedirs(f"figs/{args.exp_name}", exist_ok=True)
            np.savez(f"figs/{args.exp_name}/{args.begin}.npz", *all_qs)

        if args.plot_first:
            plt.figure(figsize=(12, 12))
            all_qs = []
            for traj_idx, key in enumerate(learner.query_state_pool.keys()):
                qs = np.array([state.q for state in learner.query_state_pool[key].states])
                plt.plot(np.arange(len(qs)) / len(qs), qs, label=traj_idx)
                all_qs.append(qs)
            plt.legend()
            plt.savefig(f"figs/{args.exp_name}/{args.begin}.png")

    if is_main_process:
        batch_num = torch.IntTensor([learner.query_batch_nums])
    else:
        batch_num = torch.IntTensor([0])

    if args.query_batch_mode == "queue":
        batch_num = args.query_batch_num_max
    else:
        if args.distribute_type == "torch":
            dist.broadcast(batch_num, src=0, async_op=False)
            batch_num = batch_num.item()
        elif args.distribute_type == "file":
            batch_num = file_system_broadcast(args.shared_file_path, batch_num, rank, world_size, var_name="batch_num")
            batch_num = batch_num.item()
        logger.info("current one contains %d batches", batch_num)

    # todo tot batch num when resuming experiment
    tot_batch_num = 0
    tot_batch_num = args.begin * batch_num * args.query_train_epoch
    for i in range(args.begin, args.end):
        if is_main_process:
            query_batch_iter = iter(learner)
            if args.query_batch_mode == "queue":
                query_batch_iter.use_wandb = args.use_wandb

        for batch_id in range(batch_num):
            if is_main_process:
                logger.info("collecting training batch %d", batch_id)
                try:
                    states_batch = next(query_batch_iter)
                    finished = torch.IntTensor([0])
                except StopIteration:
                    finished = torch.IntTensor([1])
            else:
                states_batch = None
                finished = torch.IntTensor([0])

            if batch_id < args.query_resume_batch:
                tot_batch_num += 1
            if args.query_batch_mode == "queue":
                query_batch_iter.use_wandb = args.use_wandb
                if args.distribute_type == "torch":
                    dist.broadcast(finished, src=0, async_op=False)
                elif args.distribute_type == "file":
                    finished = file_system_broadcast(args.shared_file_path, finished, rank, world_size, var_name=f"finished{batch_id}")
                if finished[0]:
                    break

            for batch_train_epoch_i in range(args.query_train_epoch):
                if batch_train_epoch_i < args.query_resume_epoch:
                    tot_batch_num += 1
                with log_execution_time(f"running cycle {i} batch {batch_id} time {batch_train_epoch_i}"):
                    run_batch(
                        args,
                        learner,
                        states_batch,
                        dpo_runner,
                        is_main_process,
                        world_size,
                        rank,
                        master_addr,
                        args.dpo_master_port,
                        worker_addr,
                        cycle_id=i,
                        batch_id=batch_id,
                        batch_epoch_idx=batch_train_epoch_i,
                        tot_batch_num=tot_batch_num,
                        plot=args.plot_fig,
                        evaluate=args.eval,
                        eval_interval=args.eval_interval
                    )
                    tot_batch_num += 1

        if is_main_process and args.update_trajs and (i + 1) % args.update_trajs_interval == 0:
            learner.gen_new_hits(logfile=f"{args.train_dataset_dir}/{args.exp_name}/{i}.log",
                                 method=args.update_trajs_method,
                                 filter_by_length=args.filter_trajs_by_length,
                                 filter_by_ans_ridx=args.filter_trajs_by_ansidx)
            learner.save_states_pool(f"{args.train_dataset_dir}/{args.exp_name}/updated_state_pool_{i}.pkl")
