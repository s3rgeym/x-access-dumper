import asyncio
import itertools
import struct
import typing
from functools import partial, wraps


def async_run(
    func: typing.Callable[..., typing.Awaitable[typing.Any]]
) -> typing.Callable:
    @wraps(func)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return asyncio.run(func(*args, **kwargs))

    return wrapper


def read_struct(fp: typing.BinaryIO, format: str) -> tuple[typing.Any, ...]:
    return struct.unpack(format, fp.read(struct.calcsize(format)))


def permutate_strings(*args: typing.Any) -> typing.Iterable[str]:
    """
    >>> list(permutate_strings(('', 'dir/'), ('file1', 'file2', 'file3'), ('.ext1', '.ext2')))
    ['file1.ext1', 'file1.ext2', 'file2.ext1', 'file2.ext2', 'file3.ext1', 'file3.ext2', 'dir/file1.ext1', 'dir/file1.ext2', 'dir/file2.ext1', 'dir/file2.ext2', 'dir/file3.ext1', 'dir/file3.ext2']
    """
    return map(partial(str.join, ''), itertools.product(*args))
