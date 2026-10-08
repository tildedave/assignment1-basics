import pickle
import time
import argparse
import regex as re
from copy import copy
from collections import Counter
from collections.abc import Iterator
from itertools import islice, pairwise, chain
from functools import partial, cached_property
from io import BytesIO
from typing import BinaryIO, TextIO
from multiprocessing import cpu_count, Pool
import os

import numpy as np

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# From Claude
def split_stream(f: TextIO, delim: str, block_size: int = 4096) -> Iterator[str]:
    buf = ""
    while block := f.read(block_size):
        buf += block
        *complete, buf = buf.split(delim)
        for piece in complete:
            yield piece
            yield delim
    yield buf


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(s: str, *, whitespace=False) -> Iterator[bytes]:
    if whitespace:
        for w in re.splititer(r"\s", s):
            yield w.encode()
        return

    it = re.finditer(PAT, s)
    for m in it:
        yield m.group(0).encode()


def test_pretokenize():
    gen = pretokenize(
        "Call me Ishmael. Some years ago- never mind how long precisely- having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation."
    )
    assert [b.decode("utf-8") for b in islice(gen, 10)] == [
        "Call",
        " me",
        " Ishmael",
        ".",
        " Some",
        " years",
        " ago",
        "-",
        " never",
        " mind",
    ]


def test_pretokenize_whitespace():
    gen = pretokenize(
        "Call me Ishmael. Some years ago- never mind how long precisely- having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.",
        whitespace=True,
    )
    assert [b.decode("utf-8") for b in islice(gen, 10)] == [
        "Call",
        "me",
        "Ishmael.",
        "Some",
        "years",
        "ago-",
        "never",
        "mind",
        "how",
        "long",
    ]


def test_pretokenize_whitespace2():
    gen = pretokenize(
        "newest newest newest newest",
        whitespace=True,
    )
    assert [b.decode("utf-8") for b in islice(gen, 100)] == [
        "newest",
        "newest",
        "newest",
        "newest",
    ]


def most_common_elements[T](c: Counter[T]) -> list[T]:
    num_look_at = 2
    elements = c.most_common(num_look_at)
    most_common_freq = elements[0][1]

    while True:
        least_common_freq = elements[-1][1]
        if least_common_freq != most_common_freq:
            return [s[0] for s in elements if s[1] == most_common_freq]

        if num_look_at >= len(c):
            return list(c.keys())

        num_look_at *= 2
        elements = c.most_common(num_look_at)


def test_most_common_elements():
    assert most_common_elements(Counter("abc")) == ["a", "b", "c"]
    assert most_common_elements(Counter("aabbc")) == ["a", "b"]
    assert most_common_elements(Counter("aaabb")) == ["a"]


DEFAULT_VOCAB = {i: bytes([i]) for i in range(0, 256)}

MergeList = list[tuple[bytes, bytes]]
Vocab = dict[int, bytes]


def apply_merge(v: bytes, merger: tuple[bytes, bytes]) -> tuple[bytes, bool]:
    for pair in pairwise(v):
        if pair == merger:
            break
    else:
        return v, False

    # reconstruct v with the merged entity
    # we'll just use list operations even though it's dumber, I'm too
    # lazy to code this right
    new_v = []
    i = 0
    while i < len(v):
        if i == len(v) - 1:
            new_v.append(v[i])
        elif (v[i], v[i + 1]) == merger:
            new_v.append(v[i] + v[i + 1])
            i += 1
        else:
            new_v.append(v[i])
        i += 1

    return tuple(new_v), True


def pretokenize_chunk(s: str, *, whitespace: bool = False):
    pretokens = pretokenize(s, whitespace=whitespace)
    chunk_frequency: Counter[tuple[bytes, ...]] = Counter()
    for t in pretokens:
        # vocab item becomes a tuple of bytes low -> 'l','o','w', each item in
        # the tuple is a bytes
        chunk_frequency[tuple(bytes([b]) for b in t)] += 1

    return chunk_frequency


def pretokenize_file_chunk(path: str | os.PathLike, split_re: str, t: tuple[int, int]):
    start, end = t
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        frequency = Counter()
        for part in re.split("|".join(split_re), chunk):
            frequency.update(pretokenize_chunk(part))

        return frequency


def bpe(
    input_path: str | os.PathLike | BytesIO,
    *,
    vocab_size,
    special_tokens=None,
    whitespace=False,
    desired_parallelism=cpu_count(),
) -> tuple[Vocab, MergeList]:
    vocab = copy(DEFAULT_VOCAB)
    next_vocab_key = max(vocab.keys()) + 1
    if not special_tokens:
        special_tokens = []
    else:
        split_re = []
        for special_token in special_tokens:
            vocab[next_vocab_key] = special_token.encode()
            next_vocab_key += 1
            split_re.append(re.escape(special_token))

    start_pretokenize = time.monotonic()
    merges = []
    if isinstance(input_path, BytesIO):
        word_frequency = pretokenize_chunk(input_path.read().decode(), whitespace=whitespace)
    else:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, desired_parallelism, b"<|endoftext|>")

        with Pool(processes=desired_parallelism) as pool:
            word_frequency: Counter[tuple[bytes, ...]] = Counter()
            for chunk_freq in pool.imap_unordered(
                partial(pretokenize_file_chunk, input_path, split_re), zip(boundaries[:-1], boundaries[1:])
            ):
                word_frequency.update(chunk_freq)

    freq: Counter[tuple[bytes, bytes]] = Counter()
    for v in word_frequency:
        for pair in pairwise(v):
            freq[pair] += word_frequency[v]
    end_pretokenize = time.monotonic()

    print("pretokenization complete in", end_pretokenize - start_pretokenize)

    recalculate_words = []

    while len(vocab) < vocab_size:
        if not freq:
            # Every word in our original text has been merged to a single token
            break

        for v in recalculate_words:
            for pair in pairwise(v):
                freq[pair] += word_frequency[v]
        recalculate_words = []

        merger = max(most_common_elements(freq))

        # merge in our vocab now
        new_frequency: Counter[bytes] = Counter()

        vocab[next_vocab_key] = merger[0] + merger[1]
        next_vocab_key += 1

        merges.append(tuple([merger[0], merger[1]]))

        for v in word_frequency:
            new_v, changed = apply_merge(v, merger)
            if changed:
                recalculate_words.append(new_v)
                for pair in pairwise(v):
                    freq[pair] -= word_frequency[v]

            new_frequency[new_v] = word_frequency[v]

        word_frequency = new_frequency

    return vocab, merges


def serialize_vocab(file_path: str | os.PathLike, vocab: Vocab):
    with open(file_path, "wb") as f:
        pickle.dump(vocab, f)


def serialize_merges(file_path: str | os.PathLike, merges: MergeList):
    with open(file_path, "wb") as f:
        pickle.dump(merges, f)


def test_bpe():
    text = BytesIO(
        b"""low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""
    )
    vocab_, merges = bpe(text, vocab_size=256 + 1 + 6, special_tokens=["<|endoftext|>"], whitespace=True)
    vocab = set(vocab_.values())

    for b in range(256):
        assert bytes([b]) in vocab
        vocab.remove(bytes([b]))

    assert vocab == frozenset([b"<|endoftext|>", b"st", b"est", b"ow", b"low", b"west", b"ne"])
    assert merges == [(b"s", b"t"), (b"e", b"st"), (b"o", b"w"), (b"l", b"ow"), (b"w", b"est"), (b"n", b"e")]
    with open("vocab-out.txt", "w") as f:
        f.write("\n".join([f"{k} -> {v}" for k, v in vocab_.items()]))
    serialize_vocab("vocab.pickle", vocab_)
    with open("merges-out.txt", "w") as f:
        f.write("\n".join([f"{m1} {m2}" for m1, m2 in merges]))
    serialize_merges("merges.pickle", vocab_)


class Tokenizer:
    special_tokens: list[str]

    def __init__(self, vocab: Vocab, merges: MergeList, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) -> "Tokenizer":
        t = Tokenizer()
        with open(vocab_filepath, "rb") as f:
            t.vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            t.merges = pickle.load(f)

        t.special_tokens = special_tokens or []
        return t

    def encode(self, text: str, *, whitespace=False) -> list[int]:
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        result = []

        if self.special_tokens:
            parts = [p for p in re.split(f"({self.split_re})", text) if p]
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                result.append(reverse_vocab[part.encode()])
                continue

            pretokens = pretokenize(part, whitespace=whitespace)
            for p in pretokens:
                p_bytes = [bytes([b]) for b in p]
                # apply the merges to the pre-tokens
                # finally output the tokens
                for merger in self.merges:
                    p_bytes, _ = apply_merge(p_bytes, merger)

                for b in p_bytes:
                    result.append(reverse_vocab[b])

        return result

    def encode_iterable(self, f: TextIO, *, whitespace=False) -> Iterator[int]:
        for chunk in split_stream(f, delim="<|endoftext|>"):
            yield from self.encode(chunk, whitespace=whitespace)

    def decode(self, ids: list[int]) -> str:
        result: bytes = b""
        for i in ids:
            result += self.vocab[i]

        return result.decode(errors="replace")

    @cached_property
    def split_re(self):
        return "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))


def test_tokenize():
    vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t", 6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
    merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
    t = Tokenizer(vocab, merges, special_tokens=None)

    assert list(t.encode("the cat ate")) == [9, 7, 1, 5, 10, 3]
    assert t.decode([9, 7, 1, 5, 10, 3]) == "the cat ate"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CS336 Assignment 1 - Tokenizer",
        description="Tokenizes a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Input filename to tokenize")
    parser.add_argument("-bpe", "--bpe", help="BPE (from run_bpe.py)")
    parser.add_argument("-o", "--out", help="Output filename")
    parser.add_argument("--chunk-size", type=int, required=False, default=100_000, help="Chunk size for writing file")

    args = parser.parse_args()

    with open(args.bpe, "rb") as f:
        bpe_obj = pickle.load(f)
        vocab = bpe_obj["vocab"]
        merges = bpe_obj["merges"]

    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    with open(args.filename) as f, open(args.out, "wb") as out_f:
        it = tokenizer.encode_iterable(f)

        chunk_it = islice(it, args.chunk_size)
        while chunk_it:
            np_chunk = np.fromiter(chunk_it, dtype=np.uint16)
            out_f.write(np_chunk.tobytes())
            print("Wrote chunk")
            chunk_it = islice(it, args.chunk_size)
