"""Launch CPU/Gloo workers on Windows without torchrun's libuv-dependent agent."""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def launch(arguments, processes=2, timeout=180):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    children = []
    try:
        for rank in range(processes):
            env = dict(
                os.environ,
                MASTER_ADDR="127.0.0.1",
                MASTER_PORT=str(port),
                RANK=str(rank),
                LOCAL_RANK=str(rank),
                WORLD_SIZE=str(processes),
                USE_LIBUV="0",
                OMP_NUM_THREADS="1",
            )
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).resolve().parents[1] / "train_distributed.py"),
                        *arguments,
                        "--strategy",
                        "ddp",
                        "--device",
                        "cpu",
                    ],
                    env=env,
                )
            )
        deadline = time.monotonic() + timeout
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("A DDP worker failed; see its traceback above")
            if time.monotonic() >= deadline:
                raise TimeoutError("CPU DDP exceeded the launch timeout")
            time.sleep(0.2)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError("CPU DDP failed")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", type=int, default=2)
    args, rest = parser.parse_known_args()
    if args.processes < 2:
        parser.error("At least two workers are required")
    launch(rest, args.processes)
