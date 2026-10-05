from cs336_basics.tokenizer import serialize_merges, serialize_vocab

from .common import FIXTURES_PATH
from .adapters import run_train_bpe


def test_train_tinystories_validation():
    input_path = FIXTURES_PATH / "TinyStories-valid.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )

    serialize_vocab(vocab)
    serialize_merges(merges)


def test_train_tinystories_full():
    input_path = FIXTURES_PATH / "TinyStories-train.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )
    serialize_vocab(vocab)
    serialize_merges(merges)
