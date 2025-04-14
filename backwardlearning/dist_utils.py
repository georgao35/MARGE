import os
import tempfile
import torch
import time
import logging


def file_system_broadcast(temp_dir, data, rank, world_size, var_name="data"):

    broadcast_file = os.path.join(temp_dir, f"broadcast_{var_name}.pt")
    # todo handle existing broadcast. when existed, add var_name with a suffix

    if rank == 0:
        # Master node writes the data to the file
        torch.save(data, broadcast_file)
        logging.debug("saved to %s", broadcast_file)
    else:
        # Non-master nodes wait for the file to appear and then read the data
        while not os.path.exists(broadcast_file):
            time.sleep(0.1)  # Wait for the master to write the file
            logging.debug("waiting for broadcast file %s", broadcast_file)

        data = torch.load(broadcast_file)
        logging.debug("load from file %s", broadcast_file)

    # Ensure all nodes have read the data before proceeding
    file_system_barrier(temp_dir, rank, world_size, f"broadcast_barrier_{var_name}")

    # if rank == 0:
        # os.remove(broadcast_file)
        # clear_barrier_file(temp_dir, f"broadcast_barrier_{var_name}", world_size)

    return data


def file_system_barrier(temp_dir, rank, world_size, barrier_name, flush: bool=True):

    barrier_file = os.path.join(temp_dir, barrier_name)
    with open(f"{barrier_file}.{rank}", "w") as f:
        f.write("1")
        if flush:
            f.flush()
            os.fsync(f.fileno())

    # Wait for all nodes to touch the barrier
    while True:
        barrier_files = [f"{barrier_file}.{i}" for i in range(world_size)]
        if all(os.path.exists(f) for f in barrier_files):
            logging.info("barrier %s finished, rank %d", barrier_name, rank)
            break
        time.sleep(0.1)  # Wait for the other nodes
        logging.debug("waiting for barrier %s", barrier_name)


def clear_barrier_file(temp_dir, barrier_name, world_size):
    for i in range(world_size):
        os.remove(os.path.join(temp_dir, f"{barrier_name}.{i}"))
