import regex as re
from copy import copy
from collections import Counter
from collections.abc import Iterator
from itertools import islice, pairwise
from itertools import chain

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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


def apply_merge(v: bytes, merger: tuple[bytes, bytes]):
    for pair in pairwise(v):
        if pair == merger:
            break
    else:
        return v

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

    return tuple(new_v)


def bpe(s: str, *, vocab_size, special_tokens=None, whitespace=False) -> tuple[Vocab, MergeList]:
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

    merges = []

    pretokens: Iterator[bytes]
    if split_re:
        pretokens = chain.from_iterable(pretokenize(part) for part in re.split("|".join(split_re), s))
    else:
        pretokens = pretokenize(s, whitespace=whitespace)

    print("pretokenization complete")
    word_frequency: Counter[tuple[bytes, ...]] = Counter()
    for t in pretokens:
        # vocab item becomes a tuple of bytes low -> 'l','o','w', each item in
        # the tuple is a bytes
        word_frequency[tuple(bytes([b]) for b in t)] += 1

    while len(vocab) < vocab_size:
        freq: Counter[tuple[bytes, bytes]] = Counter()
        for v in word_frequency:
            for pair in pairwise(v):
                freq[pair] += word_frequency[v]

        if not freq:
            # Every word in our original text has been merged to a single token
            break

        merger = max(most_common_elements(freq))

        # merge in our vocab now
        new_frequency: Counter[bytes] = Counter()

        vocab[next_vocab_key] = merger[0] + merger[1]
        next_vocab_key += 1

        merges.append(tuple([merger[0], merger[1]]))

        for v in word_frequency:
            new_frequency[apply_merge(v, merger)] = word_frequency[v]

        word_frequency = new_frequency
        print(len(vocab))

    return vocab, merges


def test_bpe():
    # this verifies that we don't run into issues with | I guess
    text = """low low low low low<|endoftext|>
lower lower widest widest widest<|endoftext|>
newest newest newest newest newest newest"""
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
            p_bytes = apply_merge(p_bytes, merger)

        for b in p_bytes:
            yield reverse_vocab[b]


def test_tokenize():
    vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t", 6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
    merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]

    assert list(tokenize("the cat ate", vocab, merges)) == [9, 7, 1, 5, 10, 3]
