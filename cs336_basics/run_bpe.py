import argparse
import pickle
import time

from cs336_basics.tokenizer import bpe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CS336 Assignment 1 - Tokenizer",
        description="Tokenizes a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Input filename")
    parser.add_argument("-vs", "--vocab-size", type=int, required=True, help="max vocab size")
    parser.add_argument("--out", required=True, help="filename to write vocab/merges to")

    args = parser.parse_args()

    start_time = time.monotonic()
    vocab, merges = bpe(args.filename, vocab_size=args.vocab_size, special_tokens=["<|endoftext|>"])
    print(
        f"completed bpe in {time.monotonic() - start_time:.2f}ms",
    )

    with open(args.out, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)

    print(f"Wrote vocab/merges to {args.out}")
