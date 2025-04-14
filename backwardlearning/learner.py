import os
import logging
import time
import random
import json
from typing import List, Tuple, Dict, Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import jsonlines
import copy
import pickle
from functools import partial
from itertools import pairwise

import wandb
import numpy as np
from tqdm.contrib import tzip, tenumerate
from vllm import SamplingParams
import matplotlib.pyplot as plt

from backwardlearning.utils import (
    State,
    get_stepbased_states,
    get_num_based_states,
    get_states_from_responses, split_states_on_query,
    launch_fastchat_server, stop_fastchat_server,
    get_states_results_concurrent,
    convert_state_hf_json,
    log_execution_time,
    save_states
)
from backwardlearning.generator import (
    APIGenerator
)
from backwardlearning.storage import (
    StatesHeap,
    QueryStates,
    get_update_node
)
from backwardlearning.dpo_runner import DPORunner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__file__)


class Learner:

    def __init__(
        self,
        args,
        queries,
        gts,
        model_name: str,
        model_path: str,
        dpo_script_name: str,
        dpo_dataset_dir: str,
        dpo_output_dir: str,
        vllm_server_url: str,
        vllm_server_num_gpus: int,
        vllm_server_post_num_threads: int=64,
        vllm_log_dir: str="logs",
        evaluation_trajs_num: int=64,
        evaluation_process_num: int=8,
        init_trajs: List[str]=None,
        init_trajs_only_correct: bool=True,
        initial_node_depth: int=-3,  # the layer from which to learn
        pruning_depth: int=-5,
        buffer_max_p: float=0.7,  # the maximum and minimum q estimate to form the buffer
        buffer_min_p: float=0.3,
        query_batch_size: int=64,
        query_batch_mode: str="iteration",
        eval_dataset: Tuple[List[str], List[str], List[str]]=None,
        query_timeout_iter: int=15,
        use_step_based: bool=False,
        vine_reinforce: bool=False,
        tokenizer = None,
    ):

        self.args = args
        self.tokenizer = tokenizer
        self.generator = APIGenerator(
            model_name,
            vllm_server_url,
            model_type=args.model_type,
            first_token_explore=args.flaming_decoding
        )
        self.model_path = model_path
        self.server_gpu_num = vllm_server_num_gpus
        self.sampling_params = SamplingParams(
            n=evaluation_trajs_num,
            temperature=float(os.environ.get("TEMP", 0.8)),
            top_p=float(os.environ.get("TOP_P", 0.95)),
            top_k=int(os.environ.get("TOP_K", -1)),
            repetition_penalty=float(os.environ.get("REP_PEN", 1.0)),
            frequency_penalty=float(os.environ.get("FREQ_PEN", 0.0)),
            max_tokens=int(os.environ.get("MAX_LEN", 2048)),
        )
        self.evaluation_process_num = evaluation_process_num

        self.dpo_dataset_dir = dpo_dataset_dir
        self.dpo_output_dir = dpo_output_dir
        os.makedirs(dpo_dataset_dir, exist_ok=True)
        os.makedirs(dpo_output_dir, exist_ok=True)
        self.dpo_runner = DPORunner(
            script_name=dpo_script_name,
            num_process=vllm_server_num_gpus
        )

        self.use_step_based_states = use_step_based
        if init_trajs is None:
            self.queries, self.gts, init_trajs = self.collect_initial_trajs(queries=queries, gts=gts, method=args.update_trajs_method, n=args.init_trajs_number)
            self.evaluate_trajs = init_trajs
            print(f"{len(self.queries)=}, {len(self.gts)=} {len(init_trajs)=}")
            with open(os.path.join(f"{args.train_dataset_dir}/{args.exp_name}", "init_trajs.jsonl"), "w") as f:
                for (query, gt, traj) in zip(self.queries, self.gts, init_trajs):
                    json.dump({"traj": traj, "query": query, "gt": gt}, f)
                    f.write("\n")
        else:
            self.queries = queries
            self.gts = gts
            self.evaluate_trajs = init_trajs
        self.query_state_pool: Dict[int, QueryStates] = {}
        self.step_states_skip = args.step_states_skip
        if vine_reinforce:
            self.final_state_pool: Dict[int, State] = {}
        self.query_name_id_map: Dict[str, int] = {}
        for query_id, (query, gt, succ_traj) in tenumerate(zip(self.queries, self.gts, init_trajs)):
            if use_step_based:
                states, _ = get_stepbased_states(
                    query, succ_traj, gt,
                    query_id=query_id, tokenizer=tokenizer,
                    step_skip=self.step_states_skip, starting=True
                )
            else:
                states, _ = get_num_based_states(
                    query, succ_traj, gt,
                    query_id=query_id, tokenizer=tokenizer,
                    num_states=self.args.states_number
                )
            self.query_state_pool[query_id] = QueryStates(query=query, query_id=query_id, states=states, state_idx=initial_node_depth)
            self.query_name_id_map[query] = query_id

            if vine_reinforce:
                self.final_state_pool[query_id] = State(
                    query=query,
                    query_id=query_id,
                    rollout_id=0,
                    state=succ_traj,
                    gt=gt,
                    gen=[succ_traj],
                    q=1. if init_trajs_only_correct else 0.
                )

        if vine_reinforce and not init_trajs_only_correct:
            get_states_results_concurrent(list(self.final_state_pool.values()), cpu_count(), 1)

        if eval_dataset is not None:
            self.eval_queries, self.eval_gts, self.eval_trajs = eval_dataset
            self.eval_query_state_pool = {}
            self.eval_query_name_id_map = {}
            for query_id, (query, gt, succ_traj) in tenumerate(zip(self.eval_queries, self.eval_gts, self.eval_trajs)):
                states, _ = get_states_from_responses(query, succ_traj, gt, num_states=0, return_terminal=True, query_id=query_id)
                self.eval_query_state_pool[query_id] = QueryStates(query=query, query_id=query_id, states=states, state_idx=initial_node_depth)
                self.eval_query_name_id_map[query] = query_id
            self.eval_sampling_params = SamplingParams(
                n=1,
                max_tokens=2048,
                temperature=0.,
                top_p=1,
                top_k=1,
            )

        self.buffer_max_p = buffer_max_p
        self.buffer_min_p = buffer_min_p
        self.pruning_depth = pruning_depth
        self.pruned_query_ids = []

        # init_states = self.get_current_nodes_pruning(list(self.query_state_pool.keys()))
        # self.evaluate_states(init_states)
        # self.initialize_states_heap(init_states)

        self.query_batch_size = query_batch_size
        self.query_batch_nums = (len(self.query_state_pool) + query_batch_size - 1) // query_batch_size
        self.query_batch_mode = query_batch_mode
        assert query_batch_mode in ["iteration", "queue"]
        self.query_timeout_iter = query_timeout_iter

        self.vine_reinforce = vine_reinforce

        logger.info("initialized learner")

    def save_states_pool(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.query_state_pool, f)

    def update_pruning_depth(self):
        pass

    def get_queries_states_by_idx(self, idx: int, keys: List[int]):
        return [self.query_state_pool[key][idx] for key in keys]

    def get_current_nodes(self):
        return [query_state.curr_state for query_id, query_state in self.query_state_pool.items()]

    def get_current_nodes_pruning(self, keys: List[int]):
        prune_nodes = self.get_queries_states_by_idx(self.pruning_depth, keys)
        self.evaluate_states(prune_nodes)
        get_states_results_concurrent(prune_nodes, cpu_count(), 1)
        self.pruned_query_ids = [node.query_id for node in prune_nodes if node.q >= self.buffer_max_p]
        return [self.query_state_pool[node.query_id].curr_state for node in prune_nodes if node.q < self.buffer_max_p]

    def initialize_states_heap(self, initial_states: List[State]):
        self.states_heap = StatesHeap(states=initial_states, buffer_max_p=self.buffer_max_p, buffer_min_p=self.buffer_min_p)

    def gen_new_hits(self, logfile: str=None, method: str="random", filter_by_length: bool=False, filter_by_ans_ridx: bool=False):
        updated_trajs = {}
        if method in ["random", "preference"]:
            for query_id, query_states in self.query_state_pool.items():
                state_0 = query_states[0]
                responses = state_0.gen
                scores = state_0.meta['scores']
                scores = np.array(scores)
                if filter_by_length:
                    ans_len = np.array(state_0.meta["gen_length"])
                    scores = scores * (ans_len < 3500)
                if filter_by_ans_ridx:
                    ans_ridx = np.array(state_0.meta["ans_ridx"])
                    scores = scores * (ans_ridx < 75)

                if method == "preference" and state_0.q > 0.51:
                    correct_responses = [res for res, score in zip(responses, scores) if not score]
                else:
                    correct_responses = [res for res, score in zip(responses, scores) if score]
                if len(correct_responses) == 0:
                    continue
                correct = random.choice(correct_responses)

                if self.use_step_based_states:
                    states, _ = get_stepbased_states(
                        state_0.query, correct, state_0.gt,
                        query_id=state_0.query_id, orig_state=state_0.state,
                        tokenizer=self.tokenizer,
                        step_skip=self.step_states_skip, starting=False
                    )
                    query_states.replace_states_from(0, states)
                    updated_trajs[query_id] = correct
                else:
                    states, _ = get_num_based_states(
                        state_0.query, correct, state_0.gt,
                        query_id=state_0.query_id, orig_state=state_0.state,
                        tokenizer=self.tokenizer, return_first=False,
                        num_states=self.args.states_number
                    )
                    query_states.replace_states_from(0, states)
                    updated_trajs[query_id] = correct

                if self.vine_reinforce:
                    self.final_state_pool[query_id] = State(
                        query=state_0.query,
                        query_id=state_0.query_id,
                        rollout_id=0,
                        state=state_0.state + correct,
                        gt=state_0.gt,
                        gen=[state_0.state + correct],
                        q=1.,
                    )
        elif method == "advantage":
            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                res = list(executor.map(get_update_node, self.query_state_pool.items()))
            # detemine update node for each query, and if needs update
            states_update: List[State] = []
            for query_id, if_update, update_idx in res:
                if if_update:
                    states_update.append((query_id, self.query_state_pool[query_id][update_idx], update_idx))
            for query_id, state, update_idx in states_update:
                responses = state.gen
                scores = np.array(state.meta['scores'])
                # extra rule based reward
                if filter_by_length:
                    ans_len = np.array(state.meta["gen_length"])
                    scores = scores * (ans_len < 3500)
                if filter_by_ans_ridx:
                    ans_ridx = np.array(state.meta["ans_ridx"])
                    scores = scores * (ans_ridx < 75)
                correct_responses = [res for res, score in zip(responses, scores) if score]
                if len(correct_responses) == 0:
                    continue
                correct = random.choice(correct_responses)

                if self.use_step_based_states:
                    states, _ = get_stepbased_states(
                        state.query, correct, state.gt,
                        query_id=state.query_id, orig_state=state.state,
                        tokenizer=self.tokenizer,
                        step_skip=self.step_states_skip, starting=False
                    )
                    self.query_state_pool[query_id].replace_states_from(update_idx, states)
                    updated_trajs[query_id] = correct
                else:
                    states, _ = get_num_based_states(
                        state.query, correct, state.gt,
                        query_id=state.query_id, orig_state=state.state,
                        tokenizer=self.tokenizer,
                        num_states=self.args.states_number
                    )
                    self.query_state_pool[query_id].replace_states_from(update_idx, states)
                    updated_trajs[query_id] = correct

                if self.vine_reinforce:
                    self.final_state_pool[query_id] = State(
                        query=state.query,
                        query_id=state.query_id,
                        rollout_id=0,
                        state=state.state + correct,
                        gt=state.gt,
                        gen=[state.state + correct],
                        q=1.,
                    )

        elif method == "max entropy":
            pass
        elif method == "max_prob":
            for query_id, query_states in self.query_state_pool.items():
                state_0 = query_states[0]
                responses = state_0.gen
                correct = random.choice(responses)

                if self.use_step_based_states:
                    states, _ = get_stepbased_states(
                        state_0.query, correct, state_0.gt,
                        query_id=state_0.query_id, orig_state=state_0.state,
                        tokenizer=self.tokenizer,
                        step_skip=self.step_states_skip, starting=False
                    )
                    query_states.replace_states_from(0, states)
                    updated_trajs[query_id] = correct
                else:
                    states, _ = get_num_based_states(
                        state_0.query, correct, state_0.gt,
                        query_id=state_0.query_id, orig_state=state_0.state,
                        tokenizer=self.tokenizer, return_first=False,
                        num_states=self.args.states_number
                    )
                    query_states.replace_states_from(0, states)
                    updated_trajs[query_id] = correct

                if self.vine_reinforce:
                    self.final_state_pool[query_id] = State(
                        query=state_0.query,
                        query_id=state_0.query_id,
                        rollout_id=0,
                        state=state_0.state + correct,
                        gt=state_0.gt,
                        gen=[state_0.state + correct]
                    )

            if self.vine_reinforce:
                get_states_results_concurrent(list(self.final_state_pool.values()), cpu_count(), 1)
        else:
            raise NotImplementedError()

        if logfile is not None:
            with open(logfile, "w") as f:
                json.dump(updated_trajs, f, indent=2)

    def collect_initial_trajs(self, queries, gts, n=8, method: str="random", filter_by_length: bool=False, filter_by_ans_ridx: bool=False, logfile=None):
        states = []
        prefix_cot = "Let's think step by step.\n\nStep 1:" if self.use_step_based_states else ""
        for query, gt in zip(queries, gts):
            states.append(State(
                query=query, rollout_id=0,
                state=prefix_cot,
                gt=gt
            ))
        succ_queries, succ_trajs, succ_gts = [], [], []
        sampling_param = copy.deepcopy(self.sampling_params)
        sampling_param.n = n
        with log_execution_time("generating initial trajs"):
            self.evaluate_states(states, custom_sampling_params=sampling_param)
            # states = self.generator.generate_from_states(states, sampling_params=sampling_param, in_place=True)

        if method in ["random", "advantage", "preference"]:
            for state in states:
                responses = state.gen
                scores = np.array(state.meta['scores'])
                # extra rule based reward
                if filter_by_length:
                    ans_len = np.array(state.meta["gen_length"])
                    scores = scores * (ans_len < 3500)
                if filter_by_ans_ridx:
                    ans_ridx = np.array(state.meta["ans_ridx"])
                    scores = scores * (ans_ridx < 75)
                
                if method == "preference" and state.q > 0.55:
                    correct_responses = [res for res, score in zip(responses, scores) if not score]
                else:
                    correct_responses = [res for res, score in zip(responses, scores) if score]

                if len(correct_responses) == 0:
                    continue
                correct = random.choice(correct_responses)

                succ_queries.append(state.query)
                succ_trajs.append(prefix_cot + correct)
                succ_gts.append(state.gt)

        elif method == "max_prob":
            for state in states:
                succ_queries.append(state.query)
                succ_trajs.append(prefix_cot + state.gen[0])
                succ_gts.append(state.gt)
        else:
            raise NotImplementedError()

        save_states(states, f"{self.args.train_dataset_dir}/{self.args.exp_name}/init_trajs.state")
        return succ_queries, succ_gts, succ_trajs

    def collect_initial_trajs_old(self, queries, gts, max_iterations: int=32, only_correct: bool=True):
        states = []
        for query, gt in zip(queries, gts):
            states.append(State(
                query=query, rollout_id=0,
                state="Let's think step by step.\n\nStep 1:",
                gt=gt
            ))
        succ_queries, succ_trajs, succ_gts = [], [], []
        sampling_param = SamplingParams(n=1, max_tokens=1024)

        with log_execution_time("generating initial trajs"):
            if only_correct:
                for i in range(max_iterations):
                    states = self.generator.generate_from_states(states, sampling_params=sampling_param, in_place=True)
                    states = get_states_results_concurrent(states, cpu_count(), 1)
                    nxt_states = []
                    for state in states:
                        if state.q > 0:
                            succ_queries.append(state.query)
                            succ_trajs.append("Let's think step by step.\n\nStep 1:" + state.gen[0])
                            succ_gts.append(state.gt)
                        else:
                            nxt_states.append(state)
                    if len(nxt_states) == 0:
                        break
                    else:
                        states = nxt_states
                logger.info("found %d successful trajectories", len(succ_queries))
            else:
                states = self.generator.generate_from_states(states, sampling_params=sampling_param, in_place=True)
                for state in states:
                    succ_queries.append(state.query)
                    succ_trajs.append("Let's think step by step.\n\nStep 1:" + state.gen[0])
                    succ_gts.append(state.gt)
        return succ_queries, succ_gts, succ_trajs

    def evaluate_states(self, states: List[State], plot: bool=False, custom_sampling_params=None):
        print(custom_sampling_params)
        with log_execution_time("evaluation: generation"):
            states = self.generator.generate_from_states(
                states,
                self.sampling_params if custom_sampling_params is None else custom_sampling_params,
                in_place=True
            )
        save_states(states, f"{self.args.train_dataset_dir}/{self.args.exp_name}/current_states.dataset")
        # parse results
        with log_execution_time("evaluation: parsing and get result"):
            states = get_states_results_concurrent(states, self.evaluation_process_num, cpu_count() // self.evaluation_process_num)
        return states

    def evaluate_all(self, cycle_id: int, run_name=""):
        queries, gts = self.queries, self.gts
        trajs = self.evaluate_trajs
        # _, _, trajs = self.collect_initial_trajs(queries=queries, gts=gts)
        # todo: update step based
        all_states, _ = get_states_from_responses(queries, trajs, gts, return_terminal=True)

        start_time = time.time()
        gen_states = self.generator.generate_from_states(all_states, self.sampling_params, in_place=True)
        end_time = time.time()
        print(f"time for generating response {end_time - start_time} seconds")
        # evaluation
        start_time = time.time()
        gen_states = get_states_results_concurrent(gen_states, cpu_count() // 8, 8)
        end_time = time.time()
        print(f"time for getting results {end_time - start_time} seconds")
        for state in gen_states:
            state.q = np.mean(state.meta['scores'])
        # plot
        query_states = split_states_on_query(gen_states)

        all_qs = [[s.q for s in states] for states in query_states.values()]
        plt.figure(figsize=(12, 12))
        for i, qs in enumerate(all_qs):
            plt.plot(np.arange(len(qs)) / len(qs), qs, label=i)
            plt.legend()
        plt.savefig(f"figs/{run_name}_{cycle_id}.png")
        np.savez(f"figs/{run_name}_{cycle_id}.npz", *all_qs)

    def gen_dpo_dataset(
        self,
        dataset_name: str,
        states: List[State],
        pairing_method: str="short",
        filter_by_length: bool=False,
        filter_by_ans_ridx: bool=False,
    ):
        if pairing_method in ["grreinforce", "reinforce"]:
            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                gen_dataset = list(executor.map(partial(convert_state_hf_json,
                                                        pairing_method=pairing_method,
                                                        filter_by_length=filter_by_length,
                                                        filter_by_ans_ridx=filter_by_ans_ridx), states))
            random.shuffle(gen_dataset)
            with jsonlines.open(dataset_name, "w") as writer:
                writer.write_all(sum(gen_dataset, []))
        else:
            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                gen_dataset = sum(
                    executor.map(partial(convert_state_hf_json, pairing_method=pairing_method), states), []
                )
            random.shuffle(gen_dataset)
            with open(dataset_name, "w", encoding='utf-8') as f:
                json.dump(gen_dataset, f, indent=4)

    def gen_vine_dataset(self, dataset_name: str, query_ids: List[int]):
        json_objs = []
        for query_id in query_ids:
            all_states = self.query_state_pool[query_id].states
            final_state = self.final_state_pool[query_id]
            advantages = []
            steps = []
            for state, nxt_state in pairwise(all_states):
                steps.append(nxt_state.state)
                advantages.append(nxt_state.q - state.q)
            steps.append(final_state.state)
            advantages.append(final_state.q - nxt_state.q)
            json_objs.append({
                "query": final_state.query,
                "prompt": all_states[0].state,
                "response": final_state.state,
                "advantage": advantages,
                "steps": steps
            })

        random.shuffle(json_objs)
        with jsonlines.open(dataset_name, "w") as writer:
            writer.write_all(json_objs)

    def run_cycle(self, cycle_id: int, evaluate_all: bool=False):
        buffer_states, queries_poped = self.states_heap.get_buffer()
        states_to_add = []
        for query in queries_poped:
            query_id = self.query_name_id_map[query]
            if self.query_state_pool[query_id].curr_idx == 0:
                continue
            states_to_add.append(self.query_state_pool[query_id].prev_state)
            self.query_state_pool[query_id].advance()

        # train dpo
        stop_fastchat_server()
        dataset_name = f"{self.dpo_dataset_dir}/hf-cycle_{cycle_id}.json"
        model_path = self.model_path if cycle_id == 0 else f"{self.dpo_output_dir}/hf-ckpt_{cycle_id-1}"
        output_path = f"{self.dpo_output_dir}/hf-ckpt_{cycle_id}"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.gen_dpo_dataset(dataset_name, buffer_states)
        self.dpo_runner.run_dpo(
            dataset=dataset_name,
            model_path=model_path,
            output_dir=output_path
        )

        # evaluate new model
        launch_fastchat_server(output_path, num_gpus=self.server_gpu_num)
        # todo: partial update
        evaled_states = self.evaluate_states(states=buffer_states + self.states_heap.all_states + states_to_add)
        self.states_heap.initialize(evaled_states)
        # evaluate all plots
        if evaluate_all:
            self.evaluate_all(cycle_id)

    def update_queries_traj(self, logfile=None, states: List[List[State]]=None):
        if states is None:
            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                res = list(executor.map(get_update_node, self.query_state_pool.items()))
        else:
            batch_states_qids = [state[0].query_id for state in states]
            batch_query_state_pool = {qid: self.query_state_pool[qid] for qid in batch_states_qids}
            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                res = list(executor.map(get_update_node, batch_query_state_pool.items()))

        # detemine update node for each query, and if needs update
        states_update: List[State] = []
        query_update_idx = {}
        for query_id, if_update, update_idx in res:
            if if_update:
                states_update.append(self.query_state_pool[query_id][update_idx])
                query_update_idx[query_id] = update_idx
        # find responses that improve the accuracy
        query_response = {}
        updated_query_ids = set()
        # at most 3 iteration
        for _ in range(3):
            logger.info("%d states needs to be updated", len(states_update))
            if len(states_update) == 0:
                break
            nxt_states = []
            for s in states_update:
                correct_idx = [i for i, correct in enumerate(s.meta["scores"]) if correct]
                if len(correct_idx) == 0:
                    response = random.choice(s.gen)
                else:
                    sampled_idx = random.choice(correct_idx)
                    response = s.gen[sampled_idx]
                query_response[s.query_id] = response
                nxt_state = get_states_from_responses(
                    s.state, responses=response, gts=s.gt,
                    num_states=1, return_terminal=False, add_first_state=False,
                    query=s.query, query_id=s.query_id
                )
                assert len(nxt_state) == 1
                nxt_states.append(nxt_state[0])
            with log_execution_time("update queries hit traj, generation"):
                self.generator.generate_from_states(nxt_states, SamplingParams(n=10), in_place=True)
            with log_execution_time("update queries hit traj, parsing"):
                nxt_states = get_states_results_concurrent(nxt_states, cpu_count() // 8, max_subprocesses=8)
            for curr, nxt in zip(states_update, nxt_states):
                if nxt.q > curr.q:
                    # states_update.remove(curr)
                    updated_query_ids.add(curr.query_id)
            states_update = [curr for curr, nxt in zip(states_update, nxt_states) if nxt.q <= curr.q]
        # update query state
        reevaluate_states = []
        for qidx in updated_query_ids:
            update_idx = query_update_idx[qidx]
            curr_state = self.query_state_pool[qidx][update_idx]
            assert curr_state is not None
            # todo: step-i based states for update trajectory
            new_states, _ = get_states_from_responses(
                curr_state.state, responses=query_response[qidx], gts=curr_state.gt,
                num_states=-1, return_terminal=True, add_first_state=False,
                query=curr_state.query, query_id=curr_state.query_id
            )
            self.query_state_pool[qidx].replace_states_from(idx=update_idx,new_states=new_states)
            reevaluate_states.extend(new_states)
        with log_execution_time("evaluating new state when replacing hits"):
            self.evaluate_states(reevaluate_states)

        if logfile is not None:
            with open(logfile, "w") as f:
                json.dump(query_response, f, indent=2)

    def __iter__(self):
        if self.query_batch_mode == "iteration":
            return QueryIterator(self.query_state_pool, self.query_batch_size)
        if self.query_batch_mode == "queue":
            return QueryManager(self.query_state_pool, self.query_batch_size, max_iterations=self.query_timeout_iter)
        raise NotImplementedError()


class QueryIterator:

    def __init__(self, queries_states, bs: int):
        self.current_batch_idx = 0
        self.bs = bs
        self.queries_states = queries_states
        self.tot_queries_num = len(queries_states)
        self.batch_nums = (len(queries_states) + bs - 1) // bs

    def __next__(self):
        if self.current_batch_idx >= self.batch_nums:
            raise StopIteration
        all_states = []
        for query_key in range(self.current_batch_idx * self.bs, (self.current_batch_idx + 1) * self.bs):
            if query_key >= self.tot_queries_num:
                break
            query_states = self.queries_states[query_key]
            all_states.append(query_states.states)
        self.current_batch_idx += 1
        return all_states


class QueryManager:

    def __init__(
        self,
        queries_states,
        batch_size: int,
        deque_value: float=0.7,
        max_iterations: int=30,
        timeout_q_thres: float=0.3
    ):
        self.queries_states = queries_states
        self.num_queries = len(queries_states)
        self.batch_size = batch_size

        self.max_iterations = max_iterations
        self.timeout_q_thres = timeout_q_thres
        self.deque_value = deque_value
        self.timeout_queries = []
        self.succ_queries = []
        # query subset
        self.current_query_ids = list(range(batch_size))
        self.lastest_queryid = batch_size - 1
        # query info, keys are query ids
        self.query_iteration = {qid: 0 for qid in self.current_query_ids}
        self.query_s0values = {qid: [] for qid in self.current_query_ids}

        self.use_wandb = True

    @property
    def exausted(self):
        return self.lastest_queryid == self.num_queries

    def __next__(self):
        logger.info("batch current queries: %s", str(self.current_query_ids))
        if len(self.current_query_ids) == 0:
            raise StopIteration
        # get all states
        all_states = []
        for query_id in self.current_query_ids:
            all_states.append(self.queries_states[query_id].states)
        # update metric & log q0 values
        for query_id in self.current_query_ids:
            self.query_iteration[query_id] += 1
            self.query_s0values[query_id].append(self.queries_states[query_id].states[0].q)
        logger.debug("query queue, visit numbers %s", str(self.query_iteration))

        succ_queries = self.batch_filter(self.success_filter)
        timeout_queries = self.batch_filter(self.timeout_filter)
        self.succ_queries.extend(succ_queries)
        self.timeout_queries.extend(timeout_queries)
        self.batch_fillup()

        if self.use_wandb:
            try:
                q_improvement_succ = [self.query_s0values[qid][-1]-self.query_s0values[qid][1] for qid in succ_queries]
                q_improvement_timeout = [self.query_s0values[qid][-1]-self.query_s0values[qid][1] for qid in timeout_queries]
            except IndexError:
                q_improvement_succ = []
                q_improvement_timeout = []
            q_improvement = q_improvement_succ + q_improvement_timeout
            wandb.log({
                "queries/successful queries": len(self.succ_queries),
                "queries/timeout queries": len(self.timeout_queries),
                "queries/value improvement mean": np.mean(q_improvement) if len(q_improvement) > 0 else 0,
                "queries/value improvement std": np.std(q_improvement) if len(q_improvement) > 0 else 0,
                "queries/succeeded value improvement mean": np.mean(q_improvement_succ) if len(q_improvement_succ) > 0 else 0,
                "queries/succeeded value improvement std": np.std(q_improvement_succ) if len(q_improvement_succ) > 0 else 0,
                "queries/timeout value improvement mean": np.mean(q_improvement_timeout) if len(q_improvement_timeout) > 0 else 0,
                "queries/timeout value improvement std": np.std(q_improvement_timeout) if len(q_improvement_timeout) > 0 else 0,
                "queries/trained queries": self.lastest_queryid
            }, commit=False)
        return all_states

    def timeout_filter(self, query_id: int):
        res = self.query_iteration[query_id] < self.max_iterations
        if not res:
            logger.debug("remove %d due to timeout", query_id)
        return res

    def success_filter(self, query_id: int):
        res = self.queries_states[query_id].states[0].q < self.deque_value
        if not res:
            logger.debug("remove %d as its query accuracy is high", query_id)
        return res

    def batch_filter(self, filter_fn: Callable[[int], bool]):
        # when false, means under this filter fn, query id needs to be removed
        remove_ids = [qid for qid in self.current_query_ids if not filter_fn(qid)]
        for remove_id in remove_ids:
            self.current_query_ids.remove(remove_id)
            # remove from other information?
        return remove_ids

    def batch_fillup(self):
        num_fillup = min(
            self.batch_size - len(self.current_query_ids),
            self.num_queries - self.lastest_queryid - 1
        )
        append_list = list(range(self.lastest_queryid + 1, self.lastest_queryid + num_fillup + 1))
        self.current_query_ids.extend(append_list)
        for qid in append_list:
            self.query_iteration[qid] = 0
            self.query_s0values[qid] = []
        self.lastest_queryid += num_fillup
