import os
import re
import random
import time
import logging
import fcntl
import subprocess
from functools import partial
from typing import List
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from itertools import zip_longest
from contextlib import contextmanager
import pickle

import numpy as np
import torch

from backwardlearning.math_utils.grader import math_equal
from backwardlearning.math_utils.parser import extract_answer_custom, extract_answer_custom_with_idx


@contextmanager
def log_execution_time(task_name):
    start_time = time.time()  # Start timer
    logging.info("begin running %s", task_name)
    try:
        yield  # Allow execution of the code block
    finally:
        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time
        logging.info("%s took %f seconds to complete.", task_name, elapsed_time)


def acquire_read_lock(file_path):
    with open(file_path, 'r') as f:
        fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock for reading
        logging.info("Read lock acquired")
        # Read the latest content
        content = f.read()
        fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock
        logging.info("Read lock released")
    return content


def acquire_write_lock(file_path, content):
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            pass
    with open(file_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock for writing
        logging.info("Write lock acquired")
        # Perform your write operations here
        f.seek(0)
        f.write(content)
        f.truncate()  # Truncate any remaining content if the new write is shorter
        fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock
        logging.info("Write lock released")


@dataclass(order=True)
class State:
    query: str=field(compare=False)
    rollout_id: int=field(compare=False)
    state: str=field(compare=False)
    gt: str=field(compare=False)
    q: float=field(default=0.0, compare=True)
    query_id: int=field(default=None, compare=False)
    gen: str=field(default=None, compare=False)
    meta: dict=field(default_factory=dict, compare=False)
    cell_id: int=field(default=-1, compare=False)
    state_token_num: int=field(default=0, compare=False)


def convert_state_hf_json(state: State, pairing_method: str="short", filter_by_length: bool=False, filter_by_ans_ridx: bool=False):
    # todo: when the number of correct the wrong answers are different; how to deal with question position
    prompt = [{
        "content": state.state,
        "role": "user"
    }]
    responses = state.gen
    scores = state.meta['scores']
    correct_responses = [res for res, score in zip(responses, scores) if score]
    false_responses = [res for res, score in zip(responses, scores) if not score]
    json_objs = []
    if pairing_method == "long":
        fill = correct_responses[-1] if len(correct_responses) < len(false_responses) else false_responses[-1]
        for chosen, rejected in zip_longest(correct_responses, false_responses, fillvalue=fill):
            json_objs.append({
                "chosen": prompt + [{"content": chosen, "role": "assistant"}],
                "rejected": prompt + [{"content": rejected, "role": "assistant"}],
                "prompt": state.state,
                "query": state.query
            })
    elif pairing_method == "short":
        for chosen, rejected in zip(correct_responses, false_responses):
            json_objs.append({
                "chosen": prompt + [{"content": chosen, "role": "assistant"}],
                "rejected": prompt + [{"content": rejected, "role": "assistant"}],
                "prompt": state.state,
                "query": state.query
            })
    elif pairing_method == "single_sample":
        if len(correct_responses) == 0 or len(false_responses) == 0:
            return []
        chosen = random.choice(correct_responses)
        for _ in range(10):
            if len(chosen) == 0:
                chosen = random.choice(correct_responses)
            if chosen[-1] == '.' and len(chosen) < 2024:
                break
            chosen = random.choice(correct_responses)
        rejected = random.choice(false_responses)
        for _ in range(10):
            if len(rejected) == 0:
                rejected = random.choice(false_responses)
            if rejected[-1] == '.' and len(rejected) < 2024:
                break
            rejected = random.choice(false_responses)
        json_objs.append({
            "chosen": prompt + [{"content": chosen, "role": "assistant"}],
            "rejected": prompt + [{"content": rejected, "role": "assistant"}],
            "prompt": state.state,
            "query": state.query
        })
    elif pairing_method == "closest_0.5":
        if len(correct_responses) == 0 or len(false_responses) == 0:
            return []
        chosen = random.choice(correct_responses)
        for _ in range(10):
            if len(chosen) == 0:
                chosen = random.choice(correct_responses)
            if chosen[-1] == '.' and len(chosen) < 2024:
                break
            chosen = random.choice(correct_responses)
        rejected = random.choice(false_responses)
        for _ in range(10):
            if len(rejected) == 0:
                rejected = random.choice(false_responses)
            if rejected[-1] == '.' and len(rejected) < 2024:
                break
            rejected = random.choice(false_responses)
        json_objs.append({
            "chosen": prompt + [{"content": chosen, "role": "assistant"}],
            "rejected": prompt + [{"content": rejected, "role": "assistant"}],
            "prompt": state.state,
            "query": state.query
        })
    elif pairing_method == "grreinforce":
        scores = np.array(scores)
        if filter_by_ans_ridx:
            ans_ridx = np.array(state.meta["ans_ridx"])
            scores = scores * (ans_ridx < 75)
        if filter_by_length:
            ans_len = np.array(state.meta["gen_length"])
            scores = scores * (ans_len < 3000)
        if scores.std() == 0:
            return []
        advantages = (scores - scores.mean()) / scores.std()
        for i in range(0, scores.shape[0] - 1, 2):
            res = responses[i:i+2]
            adv = advantages[i:i+2]
            json_objs.append({
                "query": state.query,
                "prompt": state.state,
                "advantage": adv.tolist(),
                "response": res
            })
    elif pairing_method == "reinforce":
        pass
    else:
        pass
    return json_objs


def save_states(states, filename):
    try:
        with open(filename, 'wb') as file:
            pickle.dump(states, file)
        print(f"Object successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the object: {e}")


def shuffle_in_subgroups(lst, subgroup_size=8):
    # Check if the list can be divided into subgroups of the specified size
    if len(lst) % subgroup_size != 0:
        raise ValueError("The list length is not a multiple of the subgroup size.")
    
    # Split the list into subgroups
    subgroups = [lst[i:i+subgroup_size] for i in range(0, len(lst), subgroup_size)]
    
    # Shuffle the subgroups
    random.shuffle(subgroups)
    
    # Flatten the list of shuffled subgroups
    shuffled_list = [item for subgroup in subgroups for item in subgroup]
    
    return shuffled_list


def get_stepbased_states(
    query,
    response,
    gt,
    num_states: int=-1,
    query_id: int=0,
    orig_state: str="",
    tokenizer = None,
    step_skip: int=1,
    starting: bool=True
):
    patterns = re.finditer(r"Step \d+:", response)

    # Get the start indexes of each match in the input string
    indexes = [match.start() for match in patterns]

    # Create a list of substrings starting from each index and extending until
    # just before the next index
    states = []
    for i, index in enumerate(indexes):
        if i == num_states:
            break
        if i == len(indexes) - 1:
            # For the last step number, additional processing
            pass
            # For previous steps, we add the subsequent "Step <next index>:"
        next_step_info = "Step " + str(i + 1) + ":"
        next_part = response[:index + len(next_step_info)]
        states.append(State(
            query=query,
            rollout_id=0,
            state=orig_state + next_part,
            gt=gt,
            query_id=query_id,
        ))

    if tokenizer is not None:
        for state in states:
            state.state_token_num = len(tokenizer.tokenize(state.state))
        states = [state for state in states if state.state_token_num < os.environ.get("MAX_LEN", 2048)]

    if step_skip > 1:
        if starting:
            return states[::step_skip], []
        else:
            return states[step_skip-1::step_skip], []

    return states, []


def get_num_based_states(
    query,
    response,
    gt,
    num_states: int=5,
    query_id: int=0,
    orig_state: str="",
    tokenizer = None,
    return_first: bool=True,
    max_tokens: int=os.environ.get("MAX_LEN", 2048)
):
    response = orig_state + response
    tokens = tokenizer.encode(response)

    num_states = min(num_states, len(tokens))

    # Calculate base size and remainder
    base_size = len(tokens) // num_states
    remainder = len(tokens) % num_states
    
    result = []
    start = 0
    
    for i in range(num_states-1):
        # Add one extra element to some sublists to distribute remainder
        end = start + base_size + (1 if i < remainder else 0)
        result.append(tokens[start:end])
        start = end

    states = [
        State(
            query=query,
            rollout_id=0,
            state='',
            gt=gt,
            query_id=query_id,
            state_token_num=0
    )]
    curr_tokens = []
    for state in result:
        curr_tokens.extend(state)
        states.append(State(
            query=query,
            rollout_id=0,
            state=tokenizer.decode(curr_tokens, skip_special_tokens=True),
            gt=gt,
            query_id=query_id,
            state_token_num=len(curr_tokens)
        ))
    # if tokenizer is not None:
    #     for state in states:
    #         state.state_token_num = len(tokenizer.tokenize(state.state))
    states = [state for state in states if state.state_token_num < max_tokens]

    return states if return_first else states[1:], []


def get_states_from_responses(
    prompts,
    responses,
    gts,
    num_states: int=-1,
    return_terminal: bool=False,
    add_first_state: bool=True,
    row_split_str: str="\n\n",
    query_id: int=0,
    query: str=None
):
    states, terminal_responses = [], []
    if not isinstance(prompts, list):
        prompts = [prompts]
    if not isinstance(responses, list):
        responses = [responses]
    if not isinstance(gts, list):
        gts = [gts]
    for prompt_id, (prompt, response, gt) in enumerate(zip(prompts, responses, gts)):
        state_query = prompt if query is None else query
        if add_first_state:
            states.append(State(
                query=state_query,
                rollout_id=0,
                state=prompt,
                gt=gt,
                query_id=query_id+prompt_id
            ))
        rows = [x for x in response.removeprefix(prompt).strip().split(row_split_str) if x]
        num_rows = len(rows)
        current_string = prompt
        for i, row in enumerate(rows):
            if i == num_states:
                # the number of the first states to be parsed from response and returned.
                # when -1 or larger than num_rows, return all possible states
                break
            current_string += row
            if return_terminal and i == num_rows - 1:
                terminal_responses.append(State(
                    query=state_query,
                    rollout_id=0,
                    state=current_string,
                    query_id=query_id+prompt_id,
                    gt=gt
                ))
            else:
                current_string += row_split_str
                states.append(State(
                    query=state_query,
                    rollout_id=0,
                    state=current_string,
                    gt=gt,
                    query_id=query_id+prompt_id
                ))
    return states if not return_terminal else (states, terminal_responses)


def split_states_on_query(states: List[State]):
    query_states = {}

    for state in states:
        if state.query_id in query_states:
            query_states[state.query_id].append(state)
        else:
            query_states[state.query_id] = [state]

    return query_states


def get_state_result_concurrent(state: State, max_process: int=4):
    # get the parsed answer and score parallel
    logging.debug("extracting answer")
    with ProcessPoolExecutor(max_process) as executor:
        parsed_ans = list(executor.map(
            partial(extract_answer_custom, use_last_number=True, use_choice=False),
            state.gen
        ))
    logging.debug("checking equal")
    with ProcessPoolExecutor(max_process) as executor:
        scored = list(executor.map(
            partial(math_equal, reference=state.gt, timeout=True),
            parsed_ans
        ))

    return parsed_ans, list(scored)


def get_state_result_concurrent_new(state: State, max_process: int=4):
    # get the parsed answer and score parallel
    # also return the answer's index
    logging.debug("extracting answer new")
    with ProcessPoolExecutor(max_process) as executor:
        extracted_res = list(executor.map(
            partial(extract_answer_custom_with_idx, use_last_number=True, use_choice=False),
            state.gen
        ))
    parsed_ans = [x[0] for x in extracted_res]
    ans_idx = [x[1] for x in extracted_res]
    logging.debug("checking equal")
    with ProcessPoolExecutor(max_process) as executor:
        scored = list(executor.map(
            partial(math_equal, reference=state.gt, timeout=True),
            parsed_ans
        ))

    return parsed_ans, ans_idx, list(scored)


def get_states_results_concurrent_v0(states: List[State], max_process: int=4, max_subprocesses: int=4) -> List[State]:
    logging.debug("states num %d, state parallel process %d, answer parallel process %d", len(states), max_process, max_subprocesses)
    with ProcessPoolExecutor(max_process) as executor:
        all_res = list(executor.map(
            partial(get_state_result_concurrent, max_process=max_subprocesses), states
        ))

    for state, res in zip(states, all_res):
        state.meta.update({
            "parsed_ans": res[0],
            "scores": res[1]
        })
        state.q = np.mean(res[1])

    return states


def get_states_results_concurrent(states: List[State], max_process: int=4, max_subprocesses: int=4) -> List[State]:
    logging.debug("states num %d, state parallel process %d, answer parallel process %d", len(states), max_process, max_subprocesses)
    with ProcessPoolExecutor(max_process) as executor:
        all_res = list(executor.map(
            partial(get_state_result_concurrent_new, max_process=max_subprocesses), states
        ))

    for state, res in zip(states, all_res):
        state_gen_length_all = [len(x) for x in state.gen]
        state_gen_rindex = []
        for ans_idx, ans_len in zip(res[1], state_gen_length_all):
            rindex = ans_len - ans_idx
            state_gen_rindex.append(rindex)
        state.meta.update({
            "parsed_ans": res[0],
            "ans_idx": res[1],
            "ans_ridx": state_gen_rindex,
            "scores": res[2],
            "gen_length": state_gen_length_all
        })
        state.q = np.mean(res[2])

    return states


def launch_fastchat_server(
    model_path: str,
    num_gpus: int=torch.cuda.device_count(),
    session_name: str="llm_server",
    port: int=8000,
    wait_secs: int=None,
    log_dir: str="logs"
):
    # start new tmux session
    os.system(f"tmux new-session -d -s {session_name}")
    # Create two more windows
    for i in range(num_gpus+2):
        os.system(f"tmux new-window -t {session_name}")
    # set controller
    os.system(f"tmux send-keys -t {session_name}:1 'y' C-m")
    os.system(f"tmux send-keys -t {session_name}:1 'cd {log_dir} && python -m fastchat.serve.controller --host 0.0.0.0' C-m")
    # set openai server
    os.system(f"tmux send-keys -t {session_name}:2 'y' C-m")
    os.system(f"tmux send-keys -t {session_name}:2 'cd {log_dir} && python -m fastchat.serve.openai_api_server --host localhost --port {port}' C-m")
    # set vllm worker
    for i in range(num_gpus):
        os.system(f"tmux send-keys -t {session_name}:{i + 3} 'y' C-m")
        os.system(f"tmux send-keys -t {session_name}:{i + 3} 'cd {log_dir} && CUDA_VISIBLE_DEVICES={i} python -m fastchat.serve.vllm_worker --model-path {model_path}  --port {11000 + i} --worker-address http://localhost:{11000 + i} --model-names qwen2-7b-math --host 0.0.0.0 --trust-remote-code --conv-template qwen-7b-chat' C-m")
    if wait_secs is None:
        time.sleep(100)
    else:
        time.sleep(wait_secs)


def launch_fastchat_tp(
    model_path: str,
    controller_host: str,
    worker_host: str,
    is_main_process: bool,
    num_gpus: int=torch.cuda.device_count(),
    session_name: str="llm_server",
    port: int=8000,
    wait_secs: int=100,
    log_dir: str="logs",
    vllm_tp: int=1,
):
    if num_gpus % vllm_tp != 0:
        raise NotImplementedError("Only support when divisable by vllm_tp")
    all_gpus = list(range(num_gpus))
    visible_devices_group = [all_gpus[i:i + vllm_tp] for i in range(0, len(all_gpus), vllm_tp)]
    if is_main_process:
        num_sessions = 3 + len(visible_devices_group)
        # start new tmux session
        os.system(f"tmux new-session -d -s {session_name}")
        # Create two more windows
        for i in range(num_sessions+3):
            os.system(f"tmux new-window -t {session_name}")
        # set controller
        os.system(f"tmux send-keys -t {session_name}:1 'y' C-m")
        os.system(f"tmux send-keys -t {session_name}:1 'cd {log_dir} && python -m fastchat.serve.controller --host 0.0.0.0' C-m")
        # set openai server
        os.system(f"tmux send-keys -t {session_name}:2 'y' C-m")
        os.system(f"tmux send-keys -t {session_name}:2 'cd {log_dir} && python -m fastchat.serve.openai_api_server --host localhost --port {port}' C-m")
        # set vllm worker
        for i, visible_devices in enumerate(visible_devices_group):
            os.system(f"tmux send-keys -t {session_name}:{i + 3} 'y' C-m")
            cuda_visible_devices_str = ",".join([str(i) for i in visible_devices])
            if vllm_tp > 1:
                os.system(f"tmux send-keys -t {session_name}:{i + 3} 'cd {log_dir} && CUDA_VISIBLE_DEVICES={cuda_visible_devices_str} python -m fastchat.serve.vllm_worker --model-path {model_path}  --port {11000 + i} --worker-address http://localhost:{11000 + i} --model-names qwen2-7b-math --host 0.0.0.0 --trust-remote-code --conv-template qwen-7b-chat -tp {vllm_tp}' C-m")
            elif vllm_tp == 1:
                os.system(f"tmux send-keys -t {session_name}:{i + 3} 'cd {log_dir} && CUDA_VISIBLE_DEVICES={i} python -m fastchat.serve.vllm_worker --model-path {model_path}  --port {11000 + i} --worker-address http://localhost:{11000 + i} --model-names qwen2-7b-math --host 0.0.0.0 --trust-remote-code --conv-template qwen-7b-chat' C-m")
        if wait_secs is None:
            time.sleep(100)
        else:
            time.sleep(wait_secs)
    else:
        # start new tmux session
        os.system(f"tmux new-session -d -s {session_name}")
        # Create windows and set vllm worker
        for i, visible_devices in enumerate(visible_devices_group):
            os.system(f"tmux new-window -t {session_name}")
            os.system(f"tmux send-keys -t {session_name}:{i} 'y' C-m")
            cuda_visible_devices_str = ",".join([str(i) for i in visible_devices])
            if vllm_tp > 1:
                os.system(f"tmux send-keys -t {session_name}:{i} 'cd {log_dir} && CUDA_VISIBLE_DEVICES={cuda_visible_devices_str} VLLM_WORKER_MULTIPROC_METHOD=spawn python -m fastchat.serve.vllm_worker --controller-address http://{controller_host}:21001 --model-path {model_path}  --port {11000 + i} --worker-address http://{worker_host}:{11000 + i} --model-names qwen2-7b-math --host 0.0.0.0 --trust-remote-code --conv-template qwen-7b-chat -tp {vllm_tp} --distributed-executor-backend mp' C-m")
            elif vllm_tp == 1:
                os.system(f"tmux send-keys -t {session_name}:{i} 'cd {log_dir} && CUDA_VISIBLE_DEVICES={i} python -m fastchat.serve.vllm_worker --controller-address http://{controller_host}:21001 --model-path {model_path}  --port {11000 + i} --worker-address http://{worker_host}:{11000 + i} --model-names qwen2-7b-math --host 0.0.0.0 --trust-remote-code --conv-template qwen-7b-chat' C-m")
        time.sleep(wait_secs)


def launch_fastchat_server_worker(
    model_path: str,
    controller_host: str,
    worker_host: str,
    num_gpus: int=torch.cuda.device_count(),
    session_name: str="llm_server",
    wait_secs: int=1,
    log_dir: str="logs"
):
    # start new tmux session
    os.system(f"tmux new-session -d -s {session_name}")
    # Create windows and set vllm worker
    for i in range(num_gpus):
        os.system(f"tmux new-window -t {session_name}")
        os.system(f"tmux send-keys -t {session_name}:{i} 'y' C-m")
        os.system(f"tmux send-keys -t {session_name}:{i} 'cd {log_dir} && CUDA_VISIBLE_DEVICES={i} python -m fastchat.serve.vllm_worker --controller-address http://{controller_host}:21001 --model-path {model_path}  --port {11000 + i} --worker-address http://{worker_host}:{11000 + i} --model-names qwen2-7b-math --host 0.0.0.0 --trust-remote-code --conv-template qwen-7b-chat' C-m")
    time.sleep(wait_secs)


def stop_fastchat_server(session_name="llm_server", force: bool=False):
    if force:
        os.system("pkill -9 tmux")
    else:
        os.system(f"tmux kill-session -t '{session_name}'")
        result = subprocess.run(['tmux', 'list-sessions', '-F', '#{session_name}'], capture_output=True, text=True, check=False)
        if session_name in result.stdout:
            os.system("pkill -9 tmux")


def rename_file(root_dir, old_name, new_name):
    # Walk through the directory and its subdirectories
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == old_name:
                # Construct the full file path
                old_file_path = os.path.join(dirpath, filename)
                new_file_path = os.path.join(dirpath, new_name)

                # Rename the file
                os.rename(old_file_path, new_file_path)
                logging.info("Renamed state file: %s -> %s", old_file_path, new_file_path)
