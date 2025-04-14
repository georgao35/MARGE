import os
import subprocess
import logging

import torch

logger = logging.getLogger(__file__)


class DPORunner:

    def __init__(
        self,
        script_name: str="trl/examples/scripts/dpo.py",
        num_process: int=8,
        config_file: str="trl/examples/accelerate_configs/deepspeed_zero3.yaml",
        batch_size: int=128,
        lr: float=5e-7,
        lr_scheduler_type: str="linear",
    ):
        self.num_process_per_device = num_process
        self.batch_size = batch_size
        self.commands = [
            "accelerate", "launch",
            "--config_file", config_file,
            "--num_processes", str(num_process),
            script_name,
            # "--learning_rate", str(lr),
            # "--gradient_accumulation_steps", "16",
            "--logging_steps", "1",
            "--eval_steps", "500",
            "--warmup_ratio", "0.1",
            # "--report_to wandb",
            "--bf16", "True",
            "--logging_first_step",
            "--no_remove_unused_columns",
        ]

    def run_dpo(
        self,
        dataset: str,
        model_path: str,
        output_dir: str,
        beta: float=0.1,
        num_train_epochs: int=1,
        per_device_train_batch_size: int=1,
        resume=False,
        rpo_alpha: float=None,
        ref_model: str=None,
        metric_log_file: str=None,
    ):
        commands = self.commands + [
            "--num_train_epochs", str(num_train_epochs),
            "--dataset_name", dataset,
            "--model_name_or_path", model_path,
            "--output_dir", output_dir,
            "--beta", str(beta),
            "--per_device_train_batch_size", str(per_device_train_batch_size),
            "--gradient_accumulation_steps", str(self.batch_size // self.num_process_per_device // per_device_train_batch_size)
        ]
        if resume:
            commands += ["--resume_from_checkpoint", "true"]
        if rpo_alpha is not None:
            commands += ["--rpo_alpha", str(rpo_alpha)]
        if ref_model is not None:
            commands += ["--ref_model_name", ref_model]
        if metric_log_file is not None:
            commands += ["--metric_log_file", str(metric_log_file)]
        logger.info("running dpo with command: %s", ' '.join(commands))
        subprocess.run(
            commands,
            check=False,
            env={
                # "XFORMERS_FORCE_DISABLE_TRITON": "1",
                **os.environ.copy()
            }
        )

    def run_dpo_distributed(
        self,
        dataset: str,
        model_path: str,
        output_dir: str,
        world_size: int,
        master_addr: str,
        main_process_port: int,
        rank: int,
        learning_rate: float,
        batch_size: int=128,
        beta: float=0.1,
        num_train_epochs: int=1,
        per_device_train_batch_size: int=1,
        resume: bool=False,
        rpo_alpha: float=None,
        ref_model: str=None,
        metric_log_file: str=None,
        resume_model_path: str=None,
        lr_scheduler_type: str="linear",
        torch_dtype: str="auto",
        max_length: int=2048,
        max_prompt_length: int=1024,
        method: str="qstar",
        loss_type: str="sigmoid",
        minibatch_accum: int=1,  # used for grreinforce training to accumulate pairs into training batch
        no_shuffle: bool=False,
    ):
        batch_size = batch_size * minibatch_accum
        commands = self.commands[:4] + [
            "--num_processes", str(world_size * self.num_process_per_device),
            "--num_machines", str(world_size),
            "--main_process_ip", master_addr,
            "--main_process_port", str(main_process_port),
            "--machine_rank", str(rank),
        ] + self.commands[6:] + [
            "--learning_rate", str(learning_rate),
            "--num_train_epochs", str(num_train_epochs),
            "--dataset_name", dataset,
            "--model_name_or_path", model_path,
            "--output_dir", output_dir,
            "--lr_scheduler_type", lr_scheduler_type,
            "--beta", str(beta),
            "--per_device_train_batch_size", str(per_device_train_batch_size),
            "--gradient_accumulation_steps", str(batch_size // self.num_process_per_device // per_device_train_batch_size // world_size),
            "--torch_dtype", torch_dtype,
            "--max_length", str(max_length),
            "--max_prompt_length", str(max_prompt_length),
            "--method", method,
            "--loss_type", loss_type,
            "--sequential", str(no_shuffle)
        ]
        if resume_model_path is not None:
            commands += ["--resume_from_checkpoint", resume_model_path]
        elif resume:
            commands += ["--resume_from_checkpoint", "true"]
        if rpo_alpha is not None:
            commands += ["--rpo_alpha", str(rpo_alpha)]
        if ref_model is not None:
            commands += ["--ref_model_name", ref_model]
        if metric_log_file is not None:
            commands += ["--metric_log_file", str(metric_log_file)]
        logger.info("running dpo with command: %s", ' '.join(commands))
        subprocess.run(
            commands,
            check=False,
            env={
                # "XFORMERS_FORCE_DISABLE_TRITON": "1",
                **os.environ.copy()
            }
        )
