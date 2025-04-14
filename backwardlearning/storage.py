import heapq
from typing import List, Tuple

import numpy as np

from backwardlearning.utils import State


def get_update_node(item):
    query, query_states = item
    return query, *query_states.update_idx


class QueryStates:

    def __init__(
        self,
        query: str,
        query_id: str,
        states: List[State],
        state_idx: int=None
    ):
        self.query = query
        self.query_id = query_id
        self.states = states
        self.curr_idx = len(states) - 1 if state_idx is None else state_idx
        if self.curr_idx < 0:
            self.curr_idx += len(states)

    @property
    def curr_state(self):
        return self.states[self.curr_idx]

    @property
    def prev_state(self):
        return self.states[self.curr_idx-1] if self.curr_idx >= 1  else None

    @property
    def update_idx(self):
        if len(self.states) == 1:
            return True, 0
        q = np.array([state.q for state in self.states])
        diff_q = q[1:] - q[:-1]
        diff_min, diff_argmin = diff_q.min(), diff_q.argmin()
        if not q.any():
            return True, 0
        if diff_min < 0:
            return True, diff_argmin
        return False, -1

    def replace_states_from(self, idx: int, new_states: List[State]):
        self.states = self.states[:idx+1] + new_states

    def advance(self):
        self.curr_idx -= 1 if self.curr_idx > 0 else 0

    def __len__(self):
        return len(self.states)

    def __getitem__(self, n):
        if n < -len(self.states):
            return None
        elif n < len(self.states) and n >= -len(self.states):
            return self.states[n]
        else:
            return None


class StatesHeap:

    def __init__(
        self,
        states: List[State] | None=None,
        buffer_max_p: float=0.7,
        buffer_min_p: float=0.3,
    ):
        self.max_p = buffer_max_p
        self.min_p = buffer_min_p
        if states is not None:
            self.state_heap = [(1 - state.q, state) for state in states]
            heapq.heapify(self.state_heap)
        else:
            self.state_heap = []

    def add_state(self, state: State):
        heapq.heappush(self.state_heap, (1 - state.q, state))

    def add_states(self, states: List[State]):
        for state in states:
            heapq.heappush(self.state_heap, (1 - state.q, state))

    def initialize(self, states: List[State]):
        self.state_heap = [(1 - state.q, state) for state in states]
        heapq.heapify(self.state_heap)

    def get_buffer(self, min_buffer_size: int=8) -> Tuple[List[State], List[str]]:
        _, state = self.state_heap[0]
        queries_poped = []
        buffer_states = []
        while state.q > self.min_p:
            if state.q < self.max_p:
                buffer_states.append(state)
            else:
                queries_poped.append(state.query)
            heapq.heappop(self.state_heap)
            _, state = self.state_heap[0]
        # return states that consists the buffer, and queries who needs to advance by 1
        while len(buffer_states) < min_buffer_size and len(self.state_heap) > 0:
            buffer_states.append(heapq.heappop(self.state_heap)[1])
        return buffer_states, queries_poped

    @property
    def all_states(self):
        return [item[1] for item in self.state_heap]
