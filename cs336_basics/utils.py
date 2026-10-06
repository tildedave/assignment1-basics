from collections.abc import Iterator
from typing import TextIO


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
