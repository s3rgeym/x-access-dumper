import argparse
import sys
import typing

from .dumper import XAccessDumper
from .logger import logger
from .utils import async_run


def _parse_args(argv: typing.Sequence) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('url', nargs='*', help="url or urls")
    parser.add_argument(
        '-o',
        '--output',
        help="output directory",
        default=XAccessDumper.output_directory,
    )
    parser.add_argument(
        '-H',
        '--header',
        help="additional header. Example: \"Authorization: Bearer <token>\"",
        nargs='*',
        type=lambda s: s.split(':', 2),
    )
    parser.add_argument(
        '-a',
        '--user-agent',
        help="client User-Agent",
        default=XAccessDumper.user_agent,
    )
    parser.add_argument(
        '-e',
        '--exclude-pattern',
        '--exclude',
        help="pattern for exclude files, e.g. \"\\.(ext1|ext2|ext3)$\"",
    )
    parser.add_argument(
        '--override',
        action='store_true',
        help="force override existing files",
        default=XAccessDumper.override_existing,
    )
    parser.add_argument(
        '-t',
        '--timeout',
        help="client timeout",
        type=float,
        default=XAccessDumper.timeout,
    )
    parser.add_argument(
        '--max-timeouts',
        help="max timeouts per domain",
        type=int,
        default=XAccessDumper.max_timeouts_per_domain,
    )
    parser.add_argument(
        '-w',
        '--num-workers',
        help="number of workers",
        type=int,
        default=XAccessDumper.num_workers,
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        help="be more verbose: \"-v\" - info, \"-vv\" - debug",
        default=0,
    )
    return parser.parse_args(argv)


@async_run
async def main(argv: typing.Sequence | None = None) -> int | None:
    try:
        args = _parse_args(argv)
        levels = ['WARNING', 'INFO', 'DEBUG']
        lvl = levels[min(len(levels) - 1, args.verbose)]
        logger.setLevel(lvl)
        urls = list(args.url)
        if not urls:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    break
                urls.append(line)
        dumper = XAccessDumper(
            exclude_pattern=args.exclude_pattern,
            headers=args.header,
            num_workers=args.num_workers,
            max_timeouts_per_domain=args.max_timeouts,
            override_existing=args.override,
            timeout=args.timeout,
            user_agent=args.user_agent,
        )
        await dumper.run(urls)
    except Exception as ex:
        logger.critical(ex)
        return 1
