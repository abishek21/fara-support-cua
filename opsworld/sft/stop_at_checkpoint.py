"""Stop a Linux training process after Trainer finishes writing a checkpoint."""

import argparse
import ctypes
import json
import os
import signal
import struct
from pathlib import Path


IN_CLOSE_WRITE = 0x00000008
IN_CREATE = 0x00000100
IN_MOVED_TO = 0x00000080
EVENT = struct.Struct("iIII")
REQUIRED = ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth")


def watch(directory, checkpoint, step, pid):
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    fd = libc.inotify_init1(0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1 failed")

    def add_watch(path, mask):
        descriptor = libc.inotify_add_watch(fd, os.fsencode(path), mask)
        if descriptor < 0:
            raise OSError(ctypes.get_errno(), f"inotify_add_watch failed: {path}")
        return descriptor

    try:
        directory_watch = add_watch(directory, IN_CREATE | IN_MOVED_TO)
        checkpoint_watch = None
        if checkpoint.is_dir():
            checkpoint_watch = add_watch(checkpoint, IN_CLOSE_WRITE)
        while True:
            data = os.read(fd, 4096)
            offset = 0
            while offset < len(data):
                descriptor, mask, _, name_length = EVENT.unpack_from(data, offset)
                name = data[offset + EVENT.size:offset + EVENT.size + name_length].rstrip(b"\0")
                if descriptor == directory_watch and name == os.fsencode(checkpoint.name):
                    checkpoint_watch = add_watch(checkpoint, IN_CLOSE_WRITE)
                if descriptor == checkpoint_watch and mask & IN_CLOSE_WRITE and name == b"trainer_state.json":
                    state = json.loads((checkpoint / "trainer_state.json").read_text())
                    if state["global_step"] != step or not all(
                        (checkpoint / filename).stat().st_size > 0 for filename in REQUIRED
                    ):
                        raise RuntimeError("checkpoint state is incomplete")
                    print(f"checkpoint-{step} complete; interrupting trainer PID {pid}", flush=True)
                    os.kill(pid, signal.SIGINT)
                    return
                offset += EVENT.size + name_length
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    checkpoint = args.output_dir / f"checkpoint-{args.step}"
    watch(args.output_dir, checkpoint, args.step, args.pid)


if __name__ == "__main__":
    main()