import argparse
import json
from operator import itemgetter
import pickle

import torch

from cs336_basics.model import RoPE, TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import AdamW
from cs336_basics.training_loop import load_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CS336 Assignment 1 Decoder",
        description="Generates text given an initial prompt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("checkpoint", help="checkpoint of the model to generate text from")
    parser.add_argument("--config", required=True, help="filename")
    parser.add_argument("--bpe", required=True, help="file containing BPE vocab/merges (from run_bpe.py)")
    parser.add_argument("--prompt", type=str, default="", help="default prompt")
    parser.add_argument("-d", "--device", default="mps", help="device for pytorch")
    parser.add_argument("--temperature", type=float, default="1.0", help="temperature for softmax")
    parser.add_argument("--max-tokens", type=int, required=False)
    parser.add_argument("--nucleus", type=float, default="1.0", help="hyper-parameter for top-p sampling")

    args = parser.parse_args()

    with open(args.config) as f:
        training_config = json.loads(f.read())

    d_model, num_heads, context_length, vocab_size, num_layers, d_ff, rope_theta = itemgetter(
        "d_model",
        "num_heads",
        "context_length",
        "vocab_size",
        "num_layers",
        "d_ff",
        "rope_theta",
    )(training_config)

    torch_device = torch.device(args.device)
    d_k = d_model // num_heads
    rope = RoPE(rope_theta, d_k, context_length, device=torch_device)

    # device may not matter much for decoding - test
    lm = TransformerLM(d_model, num_heads, d_ff, vocab_size, num_layers, rope, device=torch_device)
    # AdamW is unused
    load_checkpoint(args.checkpoint, lm, AdamW(lm.parameters()))

    with open(args.bpe, "rb") as f:
        bpe_obj = pickle.load(f)
        vocab = bpe_obj["vocab"]
        merges = bpe_obj["merges"]

    end_token = "<|endoftext|>"
    tokenizer = Tokenizer(vocab, merges, special_tokens=[end_token])
    end_of_text_token = next(k for k, v in vocab.items() if v == end_token.encode())
    # end_of_text_token happens to be 256 based on how we set this up

    if args.prompt:
        prompt = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.int64, device=torch.device(args.device))
    else:
        prompt = torch.tensor([], dtype=torch.int64, device=torch.device(args.device))

    with torch.no_grad():
        generated_tokens = 0
        while True:
            if args.max_tokens is not None and generated_tokens >= args.max_tokens:
                break

            distribution = lm(prompt)
            distribution /= args.temperature
            last_position = torch.exp(distribution[-1])
            probabilities = last_position / last_position.sum()

            probs, indices = torch.sort(probabilities, descending=True)
            cutoff_idx = torch.searchsorted(torch.cumsum(probs, dim=-1), args.nucleus)
            final_distribution = probabilities[0 : cutoff_idx + 1]
            print(final_distribution)
            next_token = indices[torch.multinomial(final_distribution, num_samples=1)]
            prompt = torch.cat([prompt, next_token])

            if next_token.cpu().item() == end_of_text_token:
                break

        print(tokenizer.decode(prompt.cpu().numpy()))
