from typing import IO, BinaryIO
import os
import random

import numpy.typing as npt
import torch


def data_loading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device_str: str
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_str)

    # my understanding: this function doesn't need to care about mmap mode, but
    # when we load this from disk (outside of this function) we need to care about mmaping it

    # subtract 1 so we can return predicted next token for the final element
    start_indices = [random.randint(0, len(dataset) - context_length - 1) for _ in range(batch_size)]
    slices = torch.stack(
        [
            torch.as_tensor(dataset[idx : idx + context_length + 1], device=device, dtype=torch.int64)
            for idx in start_indices
        ]
    )

    return (slices[..., :-1], slices[..., 1:])


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]
