import asyncio
import struct
import typing
from functools import wraps


def async_run(
    func: typing.Callable[..., typing.Awaitable[typing.Any]]
) -> typing.Callable:
    @wraps(func)
    def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return asyncio.run(func(*args, **kwargs))

    return wrapper


def read_struct(fp: typing.BinaryIO, format: str) -> tuple[typing.Any, ...]:
    return struct.unpack(format, fp.read(struct.calcsize(format)))
