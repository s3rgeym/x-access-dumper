from aiohttp import ClientResponse


class Error(Exception):
    message: str = "An unexpected error has occurred"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.message
        super().__init__(self.message)


class TimeoutError(Error):
    message: str = "Timeout occurred"


class BadResponse(Error):
    def __init__(self, response: ClientResponse) -> None:
        self.response = response
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"[{self.response.status}]: {self.response.request_info.url}"
