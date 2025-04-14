import os
from typing import List
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import transformers
from vllm import LLM, SamplingParams
from openai import OpenAI, OpenAIError

from backwardlearning.utils import State


class Generator:

    def __init__(
        self,
        model_name,
        eos_str: str="<|im_end|>",
        stop_str: List[str] | str="\n\n"
    ):
        self.llm = LLM(
            model=model_name
        )
        self.total_gen_tokens = 0
        self.eos_str = eos_str
        self.eos_str_len = len(eos_str)
        self.stop_str = stop_str
        self.stop_str_len = len(stop_str)

    def generate_from_states(self, states: List[State] | State, sampling_params):
        if not isinstance(states, list):
            states = [states]
        prompts = [state.state for state in states]
        outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params)
        for state, output in zip(states, outputs):
            state.gen = [o.text for o in output.outputs]
        gen_token_nums = sum([sum([len(o.token_ids) for o in output.outputs]) for output in outputs])
        self.total_gen_tokens += gen_token_nums
        return states
    
    def generate_n_steps_from_states(
        self,
        states: List[State] | State,
        sampling_params: SamplingParams,
        n_steps: int=5
    ):
        all_states, terminal_states = [], []
        prompt_states = states
        for i in range(n_steps):
            outputs = self.generate_from_states(states=prompt_states, sampling_params=sampling_params)
            prompt_states = []
            for output in outputs:
                for gen in output.gen:
                    if gen.endswith(self.stop_str):
                        all_states.append(State(
                            query=output.query,
                            cell_id=output.cell_id,
                            rollout_id=output.rollout_id,
                            state=output.state + gen,
                            gt=output.gt
                        ))
                        prompt_states.append(State(
                            query=output.query,
                            cell_id=output.cell_id,
                            rollout_id=output.rollout_id,
                            state=output.state + gen,
                            gt=output.gt,
                        ))
                    else:
                        terminal_states.append(State(
                            query=output.query,
                            cell_id=output.cell_id,
                            rollout_id=output.rollout_id,
                            state=output.state + gen,
                            gt=output.gt
                        ))
        return all_states, terminal_states

    def generate_till_eos(
        self,
        prompts: List[str] | str,
        sampling_params: SamplingParams,
    ):
        if not isinstance(prompts, list):
            prompts = [prompts]
        outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params)
        gen_responses = [[prompts[idx] + c.text for c in o.outputs] for idx, o in enumerate(outputs)]
        gen_token_nums = sum([sum([len(o.token_ids) for o in output.outputs]) for output in outputs])
        self.total_gen_tokens += gen_token_nums
        return prompts, gen_responses

    def generate_n_steps(
        self,
        prompts: List[str] | str,
        sampling_params: SamplingParams,
        n_steps: int=5
    ):
        if not isinstance(prompts, list):
            prompts = [prompts]
        gen_token_nums = 0
        states, terminal_responses = [], []
        for _ in range(n_steps):
            prev_states_num = len(states)
            outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params)
            for prompt, o in zip(prompts, outputs):
                for output in o.outputs:
                    if output.text.endswith(self.stop_str):
                        states.append(prompt + output.text)
                    else:
                        # either meets eos or reach maximum generation length
                        terminal_responses.append(prompt + output.text)
            gen_token_nums += sum([sum([len(o.token_ids) for o in output.outputs]) for output in outputs])
            prompts = states[prev_states_num:]
        self.total_gen_tokens += gen_token_nums
        return prompts, states, terminal_responses


class ServerGenerator:

    def __init__(
        self,
        model: str,
        openai_url: str,
        stop_str: List[str] | str="\n\n",
        num_parallel: int=64,
    ):
        self.model = model
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=openai_url
        )
        self.stop_str = stop_str
        self.num_parallel = num_parallel
    
    def chat_completion_thread(self, state: State, n: int):
        messages = [{
            "role": "user",
            "content": state.state
        }]

        return self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            max_tokens=2048, temperature=1.0, top_p=0.95,
            n=n
        )

    def custom_prompt_thread(self, state: State, n: int):
        prompt_template = (
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
            "<|im_start|>user\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"
            "{state}"
        )
        state_str = state.state.removeprefix(state.query)

        return self.client.completions.create(
            prompt=prompt_template.format(query=state.query, state=state_str),
            model=self.model,
            max_tokens=2048 - len(state_str.split()), temperature=0.8, top_p=0.95,
            n=n
        )

    def custom_prompt_thread_state(self, state: State, n: int):
        prompt_template = (
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
            "<|im_start|>user\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"
            "{state}"
        )
        state_str = state.state.removeprefix(state.query)

        res = self.client.completions.create(
            prompt=prompt_template.format(query=state.query, state=state_str),
            model=self.model,
            max_tokens=2048 - len(state_str.split()), temperature=0.8, top_p=0.95,
            n=n
        )

        # length = [len(choice.text) for choice in res.choices]
        print(res.choices[0].text)
        return res

    def generate_from_states(
        self,
        states: List[State] | State,
        sampling_params: SamplingParams=None,
        in_place: bool=False,
        print_generation: bool=False
    ):
        if not isinstance(states, list):
            states = [states]
        # chat_messages = [[{
        #     "role": "user",
        #     "content": state.state
        # }] for state in states]
        n = sampling_params.n
        generate_thread_fn = self.custom_prompt_thread_state if print_generation else self.custom_prompt_thread

        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            outputs = list(executor.map(
                partial(generate_thread_fn, n=n),
                states
            ))

        ret_states = []
        assert len(states) == len(outputs)
        # from ipdb import set_trace; set_trace()
        for i, (state, output) in enumerate(zip(states, outputs)):
            output_slice = output.choices
            if in_place:
                state.gen = [o.text for o in output_slice]
            else:
                ret_states.append(State(
                    query=state.query,
                    rollout_id=state.rollout_id,
                    state=state.state,
                    q=None,
                    gt=state.gt,
                    query_id=state.query_id,
                    gen=[o.text for o in output_slice],
                    meta=state.meta,
                    cell_id=state.cell_id
                ))
        # todo: add gen tokens calculation for server
        # gen_token_nums = sum([sum([len(o.token_ids) for o in output.outputs]) for output in outputs])
        # self.total_gen_tokens += gen_token_nums
        return ret_states if not in_place else states

    def generate_till_eos(
        self,
        prompts: List[str] | str,
        sampling_params: SamplingParams=None,
        concatenate_prompt_res: bool=True
    ):
        if not isinstance(prompts, list):
            prompts = [prompts]
        n = sampling_params.n
        outputs = self.client.completions.create(
            model=self.model,
            prompt=prompts,
            max_tokens=1024,
            n=n,
            temperature=1.0,
            top_p=1.0
        )
        gen_responses = [[o.text for o in outputs.choices[i*n:(i+1)*n]] for i in range(len(prompts))]
        # gen_responses = [output.text for output in outputs.choices]
        # gen_token_nums = sum([sum([len(o.token_ids) for o in output.outputs]) for output in outputs])
        # self.total_gen_tokens += gen_token_nums
        return (
            prompts,
            gen_responses,
            [[prompt + response for response in responses] for prompt, responses in zip(prompts, gen_responses)] if concatenate_prompt_res else None
        )


class APIGenerator:
    def __init__(
        self,
        model: str,
        openai_url: str,
        model_type: str="qwen",
        first_token_explore: bool=False,
    ):
        self.model = model
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=openai_url
        )

        if model_type == "qwen":
            self.prompt_template = (
                "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
                "<|im_start|>user\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"
                "{state}"
            )
        elif model_type == "qwenmath":
            self.prompt_template = (
                "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
                "<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
                "{state}"
            )
        elif model_type == "acemath":
            self.prompt_template = (
                "<|im_start|>user\n{query}\nPlease give a step-by-step answer and use a \\boxed command to denote the final answer.<|im_end|>\n<|im_start|>assistant\n"
                "{state}"
            )
        elif model_type == "llama3":
            self.prompt_template = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou're a helpful assistant<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n{state}"
            )
        elif model_type == "mistral":
            self.prompt_template = (
                "<s> [INST] You're a helpful assistant\n\n"
                "{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}. [/INST]"
                "{state}"
            )
        elif model_type == "metamath":
            self.prompt_template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
                "### Response: Let's think step by step.{state}"
            )
        else:
            raise NotImplementedError(f"{model_type=} not supported yet")
        
        self.use_first_token = first_token_explore

    def custom_prompt_thread(self, state: State, sampling_param: SamplingParams):
        state_str = state.state.removeprefix(state.query)

        temperature = sampling_param.temperature
        top_p = sampling_param.top_p
        top_k = sampling_param.top_k
        repetition_penalty = sampling_param.repetition_penalty
        frequency_penalty = sampling_param.frequency_penalty
        max_len = sampling_param.max_tokens
        n = sampling_param.n

        extra_body = {}
        if top_k != -1:
            extra_body["top_k"] = top_k
        if repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = repetition_penalty

        prefix_len = state.state_token_num if state.state_token_num > 0 else len(state_str.split())

        try:
            res = self.client.completions.create(
                prompt=self.prompt_template.format(query=state.query, state=state_str),
                model=self.model,
                max_tokens=max_len - prefix_len,
                temperature=temperature,
                top_p=top_p,
                n=n,
                frequency_penalty=frequency_penalty,
                extra_body=extra_body
            )
            return res
        except OpenAIError:
            print(self.prompt_template.format(query=state.query, state=state_str))
            return None

    def flaming_style_generate(self, state: State, sampling_param: SamplingParams):
        state_str = state.state.removeprefix(state.query)
        first_token_params = {
            "temperature": 2,
            "n": 64,
            "logprobs": 1,
            "echo": False,
            "max_tokens": 1,
            "extra_body": {
                "top_k": 20
            }
        }
        
        prompt = self.prompt_template.format(query=state.query, state=state_str)

        completion = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            **first_token_params
        )

        temperature = sampling_param.temperature
        top_p = sampling_param.top_p
        top_k = sampling_param.top_k
        repetition_penalty = sampling_param.repetition_penalty
        frequency_penalty = sampling_param.frequency_penalty
        max_len = sampling_param.max_tokens
        n = sampling_param.n

        unique_elements, counts = np.unique([str(c.text) for c in completion.choices], return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_elements = unique_elements[sorted_indices]
        arr_length = len(sorted_elements)
        if arr_length < n:
            np.random.shuffle(sorted_elements)
            repeat_count = (n + arr_length - 1) // arr_length
            extended_arr = np.tile(sorted_elements, repeat_count)
            extended_arr = extended_arr[:n]
        else:
            extended_arr = sorted_elements[:n]

        extra_body = {}
        if top_k != -1:
            extra_body["top_k"] = top_k
        if repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = repetition_penalty
        prefix_len = state.state_token_num if state.state_token_num > 0 else len(state_str.split())

        new_completions = []
        for token in extended_arr:
            completion = self.client.completions.create(
                model=self.model,
                prompt=prompt + token,
                max_tokens=max_len - prefix_len - 1,
                temperature=temperature,
                top_p=top_p,
                n=n,
                frequency_penalty=frequency_penalty,
                extra_body=extra_body
            )
            new_completions.append(token + completion.choices[0].text)
        return new_completions

    def generate_from_states(
        self,
        states: List[State] | State,
        sampling_params: SamplingParams,
        in_place: bool=False,
        num_parallel: int=128,
    ):
        if not isinstance(states, list):
            states = [states]

        if self.use_first_token:
            generate_thread_fn = self.flaming_style_generate
        else:
            generate_thread_fn = self.custom_prompt_thread

        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            outputs = list(executor.map(
                partial(generate_thread_fn, sampling_param=sampling_params),
                states
            ))

        ret_states = []
        assert len(states) == len(outputs)
        false_states = []
        for i, (state, output) in enumerate(zip(states, outputs)):
            if output is None:
                false_states.append(state)
                continue
            if self.use_first_token:
                generated = output
            else:
                output_slice = output.choices
                generated = [o.text for o in output_slice]
            if in_place:
                state.gen = generated
            else:
                ret_states.append(State(
                    query=state.query,
                    rollout_id=state.rollout_id,
                    state=state.state,
                    q=None,
                    gt=state.gt,
                    query_id=state.query_id,
                    gen=generated,
                    meta=state.meta,
                    cell_id=state.cell_id,
                    state_token_num=state.state_token_num
                ))
        # todo: add gen tokens calculation for server
        # gen_token_nums = sum([sum([len(o.token_ids) for o in output.outputs]) for output in outputs])
        # self.total_gen_tokens += gen_token_nums
        for state in false_states:
            states.remove(state)
        return ret_states if not in_place else states
