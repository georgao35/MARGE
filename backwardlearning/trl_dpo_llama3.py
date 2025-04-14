# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import multiprocessing
import os
import json
from contextlib import nullcontext
from dataclasses import dataclass, field
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.deepspeed import deepspeed_load_checkpoint
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    DPOConfig,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser
from trl.env_utils import strtobool

from backwardlearning.stepdpo_trainer import StepDPOTrainer, QStarTrainer
from backwardlearning.trainer.grreinforce_trainer import GroupRelativeReinforceTrainer, GRReinforceTrainer
from backwardlearning.trainer.vinereinforce_trainer import VineTrainer

TRL_USE_RICH = strtobool(os.getenv("TRL_USE_RICH", "0"))

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


@dataclass
class CustomDPOScriptArguments(DPOScriptArguments):
    ref_model_name: str = field(default=None, metadata={"help": "the name for reference model. if none, will be model_name_or_path"})
    metric_log_file: str = field(default=None, metadata={"help": "This field specifies the file to store metrics. default to None, which means do not save"})
    method: str = field(default="pair_grpo")
    sequential: bool = field(default=False, metadata={"help": "whether to use sequential training"})


if __name__ == "__main__":
    parser = TrlParser((CustomDPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    print(f"{args=}")
    print(f"{training_args=}")
    print(f"{model_config=}")

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Dataset
    ################

    def qstar_process(row):
        prompt_template = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're a helpful assistant<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n{state}"
        )
        prompt = prompt_template.format(query=row["query"], state=row["prompt"].removeprefix(row["query"]))
        row["prompt"] = prompt
        row["chosen"] = row["chosen"][-1]['content'] + "<|eot_id|>"
        row["rejected"] = row["rejected"][-1]['content'] + "<|eot_id|>"
        return row

    def grreinforce_process(row):
        prompt_template = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're a helpful assistant<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n{state}"
        )
        prompt = prompt_template.format(query=row["query"], state=row["prompt"].removeprefix(row["query"]))
        row["prompt"] = prompt
        row["chosen"] = row["response"][0] + "<|eot_id|>"
        row["rejected"] = row["response"][1] + "<|eot_id|>"
        row["chosen_adv"] = row["advantage"][0]
        row["rejected_adv"] = row["advantage"][1]
        return row

    def vine_reinforce_process(row):
        prompt_template = (
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
            "<|im_start|>user\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"
            "{state}"
        )
        prompt = prompt_template.format(query=row["query"], state=row["prompt"].removeprefix(row["query"]))
        row["prompt"] = prompt
        row["chosen"] = row["response"]
        row["rejected"] = ""
        # row["steps"] = 
        # row["advantage"] = row["advantage"]
        return row

    def standard_dpo_process(row):
        prompt_template = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're a helpful assistant<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{quersy}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>"
        )
        row["chosen"] = "<|start_header_id|>assistant<|end_header_id|>\n\n" + row["prompt"].removeprefix(row["query"]) + row["chosen"][-1]['content'] + "<|eot_id|>"
        row["rejected"] = "<|start_header_id|>assistant<|end_header_id|>\n\n" + row["prompt"].removeprefix(row["query"]) + row["rejected"][-1]['content'] + "<|eot_id|>"
        prompt = prompt_template.format(query=row["query"])
        row["prompt"] = prompt
        return row

    def step_dpo_process(example):
        text_chosen = example['chosen']
        text_rejected = example['rejected']

        prompt_input = None
        prompt_no_input = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        if len(example['initial_reason_steps']) == 0:
            new_example = {
                'prompt': prompt_no_input.format(instruction=example['prompt']),
                'chosen': text_chosen,
                'rejected': text_rejected,
            }
        else:
            new_example = {
                'prompt': prompt_no_input.format(instruction=example['prompt']) + example['initial_reason_steps'],
                'chosen': text_chosen,
                'rejected': text_rejected,
            }

        return new_example

    if args.method == "qstar" or args.method == "pair_grpo":
        ds = load_dataset("json", data_files=args.dataset_name.split(','))
        if args.sanity_check:
            for key in ds:
                ds[key] = ds[key].select(range(50))
        ds = ds.map(
            qstar_process,
            # num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=False,
        )
        # ds = ds.train_test_split(test_size=0.02)
        # train_dataset = ds[args.dataset_train_split]
        # eval_dataset = ds[args.dataset_test_split]
    elif args.method == "dpo":
        ds = load_dataset("json", data_files=args.dataset_name.split(','))
        if args.sanity_check:
            for key in ds:
                ds[key] = ds[key].select(range(50))
        ds = ds.map(
            standard_dpo_process,
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=False,
        )
    elif args.method == "stepdpo":
        if ".json" in args.dataset_name:
            ds = load_dataset(
                "json",
                data_files=args.dataset_name.split("||"),
            )
        else:
            ds = load_dataset(args.dataset_name)
        ds = ds.map(
            step_dpo_process,
            num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=False,
        )
    elif args.method == "grreinforce":
        ds = load_dataset("json", data_files=args.dataset_name.split(','))
        if args.sanity_check:
            for key in ds:
                ds[key] = ds[key].select(range(50))
        ds = ds.map(
            grreinforce_process,
            # num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=False,
        )
    elif args.method == "vine_reinforce":
        ds = load_dataset("json", data_files=args.dataset_name.split(','))
        if args.sanity_check:
            for key in ds:
                ds[key] = ds[key].select(range(50))
        ds = ds.map(
            vine_reinforce_process,
            # num_proc=multiprocessing.cpu_count(),
            load_from_cache_file=False,
        )


    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        if args.ref_model_name is not None:
            model_ref = AutoModelForCausalLM.from_pretrained(args.ref_model_name, **model_kwargs)
        else:
            model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    if args.method == "qstar" or args.method == "dpo":
        with init_context:
            trainer = QStarTrainer(
                model,
                model_ref,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=ds["test"] if "test" in ds.keys() else None,
                tokenizer=tokenizer,
                peft_config=peft_config,
                callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            )
    elif args.method == "pair_grpo":
        with init_context:
            trainer = GroupRelativeReinforceTrainer(
                model,
                model_ref,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=ds["test"] if "test" in ds.keys() else None,
                tokenizer=tokenizer,
                peft_config=peft_config,
                callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            )
    elif args.method == "stepdpo":
        for index in range(3, 5):
        # for index in random.sample(range(len(ds["train"])), 3):
            print(f"Prompt sample {index} of the raw training set:\n\n{ds['train'][index]['prompt']}")
            print(f"Chosen sample {index} of the raw training set:\n\n{ds['train'][index]['chosen']}")
            print(f"Rejected sample {index} of the raw training set:\n\n{ds['train'][index]['rejected']}")
        with init_context:
            trainer = StepDPOTrainer(
                model,
                model_ref,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=ds["test"] if "test" in ds.keys() else None,
                tokenizer=tokenizer,
                peft_config=peft_config,
                callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            )
    elif args.method == "grreinforce":
        with init_context:
            trainer = GRReinforceTrainer(
                model,
                model_ref,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=ds["test"] if "test" in ds.keys() else None,
                tokenizer=tokenizer,
                peft_config=peft_config,
                callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            )
    elif args.method == "vine_reinforce":
        with init_context:
            trainer = VineTrainer(
                model,
                model_ref,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=ds["test"] if "test" in ds.keys() else None,
                tokenizer=tokenizer,
                peft_config=peft_config,
                callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            )

    # if training_args.resume_from_checkpoint is not None:
    #     resume_from_checkpoint = get_last_checkpoint(training_args.resume_from_checkpoint)
    #     if trainer.is_deepspeed_enabled:
    #         deepspeed_load_checkpoint(
    #             trainer.model_wrapped, resume_from_checkpoint, load_module_strict=True
    #         )
    #     elif is_sagemaker_mp_enabled() or trainer.is_fsdp_enabled:
    #         trainer._load_from_checkpoint(training_args.resume_from_checkpoint, trainer.model_wrapped)

    #     # Check if saved optimizer or scheduler states exist
    #     trainer._load_optimizer_and_scheduler(training_args.resume_from_checkpoint)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    with save_context:
        trainer.save_state()
        trainer.save_model(training_args.output_dir)
        if args.metric_log_file is not None:
            with open(args.metric_log_file, "w") as f:
                json.dump(trainer.stored_metrics_logs, f, indent=0)
