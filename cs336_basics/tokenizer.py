import time

import regex as re
from copy import copy
from collections import Counter
from collections.abc import Iterator
from itertools import islice, pairwise
from functools import partial
from io import BytesIO
from typing import BinaryIO
from multiprocessing import cpu_count, Pool

import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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


def test_bpe():
    text = BytesIO(
        b"""low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""
    )
    vocab, merges = bpe(text, vocab_size=256 + 1 + 6, special_tokens=["<|endoftext|>"], whitespace=True)
    vocab = set(vocab.values())

    for b in range(256):
        assert bytes([b]) in vocab
        vocab.remove(bytes([b]))

    assert vocab == frozenset([b"<|endoftext|>", b"st", b"est", b"ow", b"low", b"west", b"ne"])
    assert merges == [(b"s", b"t"), (b"e", b"st"), (b"o", b"w"), (b"l", b"ow"), (b"w", b"est"), (b"n", b"e")]


def tokenize(s: str, vocab: dict[int, bytes], merges: MergeList, *, whitespace=False) -> Iterator[bytes]:
    pretokens = pretokenize(s, whitespace=whitespace)
    reverse_vocab = {v: k for k, v in vocab.items()}

    for p in pretokens:
        p_bytes = [bytes([b]) for b in p]
        # apply the merges to the pre-tokens
        # finally output the tokens
        for merger in merges:
            p_bytes, _ = apply_merge(p_bytes, merger)

        for b in p_bytes:
            yield reverse_vocab[b]


def test_tokenize():
    vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t", 6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
    merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]

    assert list(tokenize("the cat ate", vocab, merges)) == [9, 7, 1, 5, 10, 3]
