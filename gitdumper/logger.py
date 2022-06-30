import logging
from enum import Enum

CSI = '\033['

Color = Enum(
    'Color', 'BLACK RED GREEN YELLOW BLUE MAGENTA CYAN WHITE', start=30
)


class AnsiColorHandler(logging.StreamHandler):
    LOGLEVEL_COLORS = {
        'DEBUG': Color.BLUE,
        'INFO': Color.GREEN,
        'WARNING': Color.RED,
        'ERROR': Color.RED,
        'CRITICAL': Color.RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        message: str = super().format(record)
        # use colors in tty
        if self.stream.isatty() and (
            color := self.LOGLEVEL_COLORS.get(record.levelname)
        ):
            message = f'{CSI}{color.value}m{message}{CSI}0m'
        return message


class Logger(logging.Logger):
    def __init__(self, name: str) -> None:
        super().__init__(name, logging.DEBUG)
        console = AnsiColorHandler()
        formatter = logging.Formatter("%(levelname)-8s - %(message)s")
        console.setFormatter(formatter)
        self.addHandler(console)


logger = Logger(__name__)
