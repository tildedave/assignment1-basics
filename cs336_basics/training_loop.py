from typing import IO, BinaryIO
import os
import random
import argparse

from cs336_basics.model import TransformerLM, RoPE
from cs336_basics.training import AdamW, cross_entropy

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from jaxtyping import Int, Float


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device_str: str
) -> tuple[Int[Tensor, " batch_size sequence_length"], Int[Tensor, " batch_size sequence_length"]]:
    device = torch.device(device_str)

    # my understanding: this function doesn't need to care about mmap mode, but
    # when we load this from disk (outside of this function) we need to care about mmaping it

    # subtract 1 so we can return predicted next token for the final element
    start_indices = [random.randint(0, len(dataset) - context_length - 1) for _ in range(batch_size)]
    slices = torch.stack(
        [
            # copy() suppresses a writability warning in numpy
            torch.as_tensor(dataset[idx : idx + context_length + 1].copy(), device=device, dtype=torch.int64)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CS336 Assignment 1",
        description="Runs a training loop on a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="filename containing a numpy array of token positions")
    parser.add_argument("-dm", "--d-model", type=int, required=True, help="model dimensions")
    parser.add_argument("-dff", "--d-ff", type=int, required=True, help="feed-forward dimension")
    parser.add_argument("-vs", "--vocab-size", type=int, required=True, help="max vocab size")
    parser.add_argument("-cl", "--context-length", type=int, required=True, help="context length")
    parser.add_argument(
        "-heads", "--heads", required=True, type=int, help="number of heads - must be divisible by model dimensions"
    )
    parser.add_argument("-l", "--num-layers", type=int, required=True, help="number of layers for the TransformerLM")

    parser.add_argument("-d", "--device", default="cpu", help="device for pytorch")
    parser.add_argument("-b1", "--beta1", type=float, default=0.9, help="beta1 hyperparameter")
    parser.add_argument("-b2", "--beta2", type=float, default=0.999, help="beta2 hyperparameter")
    parser.add_argument("-lr", "--lr", type=float, default=1e-3, help="alpha hyperparameter")
    parser.add_argument("-wd", "--weight-decay", type=float, default=0.01, help="weight decay hyperparameter")
    parser.add_argument("-rt", "--rope-theta", type=float, default=10000, help="rope theta argument")
    parser.add_argument("-bs", "--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("-ts", "--training-steps", type=int, default=5_000, help="batch size")

    args = parser.parse_args()

    # TODO: allowing passing checkpoint file [ez]

    d_k = args.d_model // args.heads
    rope = RoPE(args.rope_theta, d_k, args.context_length, device=torch.device(args.device))
    lm = TransformerLM(
        args.d_model, args.heads, args.d_ff, args.vocab_size, args.num_layers, rope, device=torch.device(args.device)
    )
    opt = AdamW(lm.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    with open(args.filename, "rb") as f:
        data = np.memmap(f, dtype=np.uint16, mode="r")

        for i in range(args.training_steps):  # TODO - need to take this from command line params
            print(f"Training Step #{i} -")
            batch, target = get_batch(data, args.batch_size, args.context_length, device_str=args.device)
            result: Float[Tensor, " batch_size sequence_length vocab_size"] = lm.forward(batch)
            loss = cross_entropy(result, target)
            print("I loss!", loss.cpu().item())
            loss.backward()
            # TODO: update learning rate - note i is 0-indexed
            opt.step()
