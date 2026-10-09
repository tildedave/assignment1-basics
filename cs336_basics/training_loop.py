from typing import IO, BinaryIO
import os
import random
import json
import argparse
import time

from cs336_basics.model import TransformerLM, RoPE
from cs336_basics.training import AdamW, cross_entropy, perplexity, gradient_clipping

import coolname
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from jaxtyping import Int, Float


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device_str: str
) -> tuple[Int[Tensor, " batch_size sequence_length"], Int[Tensor, " batch_size sequence_length"]]:
    # my understanding: this function doesn't need to care about mmap mode, but
    # when we load this from disk (outside of this function) we need to care about mmaping it

    # subtract 1 so we can return predicted next token for the final element
    start_indices = [random.randint(0, len(dataset) - context_length - 1) for _ in range(batch_size)]
    slices = torch.as_tensor(
        np.stack([dataset[idx : idx + context_length + 1] for idx in start_indices]),
        device=torch.device(device_str),
        dtype=torch.int64,
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
    parser.add_argument("checkpoint_dir", help="directory to write checkpoints into")

    parser.add_argument("--config-file", type=str, required=False, help="config file to read parameters from")

    # If not specified in config file, need the following:
    parser.add_argument("-dm", "--d-model", type=int, required=False, help="model dimensions")
    parser.add_argument("-dff", "--d-ff", type=int, required=False, help="feed-forward dimension")
    parser.add_argument("-vs", "--vocab-size", type=int, required=False, help="max vocab size")
    parser.add_argument("-cl", "--context-length", type=int, required=False, help="context length")
    parser.add_argument(
        "--num-heads", required=False, type=int, help="number of heads - must be divisible by model dimensions"
    )
    parser.add_argument("-l", "--num-layers", type=int, required=False, help="number of layers for the TransformerLM")

    parser.add_argument("-d", "--device", default="cpu", help="device for pytorch")
    parser.add_argument("-b1", "--beta1", type=float, default=0.9, help="beta1 hyperparameter")
    parser.add_argument("-b2", "--beta2", type=float, default=0.999, help="beta2 hyperparameter")
    parser.add_argument("-lr", "--lr", type=float, default=1e-3, help="alpha hyperparameter")
    parser.add_argument("-wd", "--weight-decay", type=float, default=0.01, help="weight decay hyperparameter")
    parser.add_argument("-rt", "--rope-theta", type=float, default=10000, help="rope theta argument")
    parser.add_argument("-bs", "--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("-ts", "--training-steps", type=int, default=5_000, help="number of training steps")
    parser.add_argument("-mg", "--max-gradient", type=float, default=1.0, help="max gradient value (for clipping)")
    parser.add_argument("--checkpoint-frequency", type=int, default=100, help="frequency to checkpoint")
    parser.add_argument("--restore-from-checkpoint", type=str, help="checkpoint file to restore from")

    args = parser.parse_args()

    # TODO: allowing passing checkpoint file [ez]
    torch_device = torch.device(args.device)

    if args.config_file:
        with open(args.config_file) as f:
            training_config = json.loads(f.read())
    else:
        training_config = {
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "context_length": args.context_length,
            "vocab_size": args.vocab_size,
            "num_layers": args.num_layers,
            "d_ff": args.d_ff,
            "rope_theta": args.rope_theta,
            "lr": args.lr,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "weight_decay": args.weight_decay,
            "max_gradient": args.max_gradient,
        }

    from operator import itemgetter

    d_model, num_heads, context_length, vocab_size, num_layers, d_ff, rope_theta = itemgetter(
        "d_model",
        "num_heads",
        "context_length",
        "vocab_size",
        "num_layers",
        "d_ff",
        "rope_theta",
    )(training_config)
    lr, beta1, beta2, weight_decay, max_gradient = itemgetter("lr", "beta1", "beta2", "weight_decay", "max_gradient")(
        training_config
    )

    d_k = d_model // num_heads
    rope = RoPE(rope_theta, d_k, context_length, device=torch_device)

    lm = TransformerLM(d_model, num_heads, d_ff, vocab_size, num_layers, rope, device=torch_device)
    if args.device == "mps":
        lm = torch.compile(lm, backend="aot_eager")
    elif args.device == "cpu":
        lm = torch.compile(lm)

    opt = AdamW(lm.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    num_tokens = 0
    num_iterations = 0
    start_iteration = 0

    if args.restore_from_checkpoint:
        num_iterations = load_checkpoint(args.restore_from_checkpoint, lm, opt)
        start_iteration = num_iterations
        run_name, _ = os.path.basename(args.restore_from_checkpoint).split(".")
        print(f"Resuming run {run_name} at iteration {num_iterations}:\n\tconfig: {training_config}")
    else:
        run_name = coolname.generate_slug()
        config_filename = os.path.join(args.checkpoint_dir, f"{run_name}.config.json")
        with open(config_filename, "w") as f:
            f.write(json.dumps(training_config))
            print(f"Saved config to {config_filename}")
        print(f"Beginning run {run_name}:\n\tconfig: {training_config} saved to {config_filename}")

    with open(args.filename, "rb") as f:
        data = np.memmap(f, dtype=np.uint16, mode="r")

        while num_iterations < args.training_steps:
            if num_iterations != start_iteration and num_iterations % args.checkpoint_frequency == 0:
                filename = os.path.join(args.checkpoint_dir, f"{run_name}.{num_iterations}")
                save_checkpoint(lm, opt, num_iterations, filename)
                print(f"Saved checkpoint {filename}")

            opt.zero_grad()
            batch, target = get_batch(data, args.batch_size, context_length, device_str=args.device)
            num_tokens += batch.numel()
            result: Float[Tensor, " batch_size sequence_length vocab_size"] = lm(batch)
            loss = cross_entropy(result, target)

            if num_iterations % 10 == 0:
                print(
                    f"Step {num_iterations}: time={time.time()}, loss={loss.cpu().item()}, perplexity={perplexity(result, target).cpu().item()}"
                )
            loss.backward()

            gradient_clipping(lm.parameters(), max_gradient)
            opt.step()

            num_iterations += 1

    filename = os.path.join(args.checkpoint_dir, f"{run_name}.final")
    save_checkpoint(lm, opt, num_iterations, filename)
    print(f"Saved final checkpoint {filename}")
