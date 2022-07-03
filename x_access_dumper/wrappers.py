import cgi
import dataclasses
import typing
from functools import cached_property

import aiohttp


@dataclasses.dataclass
class ResponseWrapper:
    _response: aiohttp.ClientResponse

    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self._response, name)

    @cached_property
    def content_type(self) -> str:
        return cgi.parse_header(self.headers.get('content-type', ''))[0]
