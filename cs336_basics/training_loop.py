import numpy as np
import numpy.typing as npt
import torch
import random


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
