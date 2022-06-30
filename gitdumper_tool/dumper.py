__all__ = ('GitDumper',)

import asyncio
import cgi
import dataclasses
import io
import re
import typing
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote, urljoin

import aiohttp
from aiohttp.typedefs import LooseHeaders

from .logger import logger
from .utils import read_struct

COMMON_FILES = [
    'COMMIT_EDITMSG',
    'FETCH_HEAD',
    'HEAD',
    'ORIG_HEAD',
    'config',
    'description',
    'index',
    'info/exclude',
    'logs/HEAD',
    'objects/info/packs',
    'packed-refs',
]

SHA1_RE = re.compile(r'\b[a-f\d]{40}\b')
PACK_RE = re.compile(r'\bpack-' + SHA1_RE.pattern[2:])
REF_RE = re.compile(r'\brefs/\S+')
SHA1_OR_REF_RE = re.compile(
    '(?P<sha1>' + SHA1_RE.pattern + ')|(?P<ref>' + REF_RE.pattern + ')'
)


@dataclasses.dataclass
class GitDumper:
    headers: LooseHeaders | None = None
    num_workers: int = 50
    output_directory: str | Path = 'output'
    override_existing: bool = False
    timeout: float = 15.0
    user_agent: str = "Mozilla/5.0 (compatible; YandexBot/3.0; +http://yandex.com/bots) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.268"

    def __post_init__(self) -> None:
        # self.num_workers = num_workers or self.num_workers
        for field in dataclasses.fields(self):
            if (
                not isinstance(field.default, dataclasses._MISSING_TYPE)
                and getattr(self, field.name) is None
            ):
                setattr(self, field.name, field.default)
        if isinstance(self.output_directory, str):
            self.output_directory = Path(self.output_directory)
        if not isinstance(self.timeout, aiohttp.ClientTimeout):
            self.timeout = aiohttp.ClientTimeout(self.timeout)

    def normalize_url(self, url: str) -> str:
        if '://' not in url:
            url = f'http://{url}'
        for x in ('/', '/.git/'):
            if not url.endswith(x):
                url += x
        return url

    async def run(self, urls: typing.Sequence[str]) -> None:
        download_queue = asyncio.Queue()
        normalized_urls = list(map(self.normalize_url, urls))
        for file in COMMON_FILES:
            for git_url in normalized_urls:
                download_url = urljoin(git_url, file)
                download_queue.put_nowait(download_url)

        # Посещенные ссылки
        seen_urls = set()

        # Запускаем задания в фоне
        workers = [
            asyncio.create_task(self.worker(download_queue, seen_urls))
            for _ in range(self.num_workers)
        ]

        # Ждем пока очередь станет пустой
        await download_queue.join()

        # Останавливаем задания
        for _ in range(self.num_workers):
            await download_queue.put(None)

        for w in workers:
            await w

        for git_url in normalized_urls:
            await self.retrieve_source_code(self.get_download_path(git_url))

    async def worker(
        self, download_queue: asyncio.Queue, seen_urls: set[str]
    ) -> None:
        async with self.get_session() as session:
            while True:
                try:
                    download_url = await download_queue.get()

                    if download_url is None:
                        break

                    if download_url in seen_urls:
                        logger.debug("already seen %s", download_url)
                        continue

                    seen_urls.add(download_url)

                    # "https://example.org/Old%20Site/.git/index" -> "output/example.org/Old Site/.git/index"
                    download_path = self.get_download_path(download_url)

                    if self.override_existing or not download_path.exists():
                        try:
                            await self.download_file(
                                session, download_url, download_path
                            )
                        except Exception as e:
                            if isinstance(e, aiohttp.ClientResponseError):
                                logger.error(
                                    "%d: %s - %s",
                                    e.status,
                                    e.message,
                                    e.request_info.url,
                                )
                            else:
                                logger.error("error: %s", e)
                            if download_path.exists():
                                logger.debug("delete: %s", download_path)
                                download_path.unlink()
                            continue
                    else:
                        logger.debug("file exists: %s", download_path)

                    await self.parse(
                        download_path, download_url, download_queue
                    )
                except Exception as ex:
                    logger.error("an unexpected error has occurred: %s", ex)
                finally:
                    download_queue.task_done()

    def get_download_path(self, download_url: str) -> Path:
        return self.output_directory.joinpath(
            unquote(download_url.split('://')[1])
        )

    async def retrieve_source_code(self, git_path: Path) -> None:
        cmd = f"git --git-dir='{git_path}' --work-tree='{git_path.parent}' checkout -- ."
        logger.debug("run: %r", cmd)
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode == 0:
            logger.info("source retrieved: %s", git_path)
        else:
            logger.error(stderr.decode())

    @asynccontextmanager
    async def get_session(self) -> typing.AsyncIterable[aiohttp.ClientSession]:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(verify_ssl=False),
            headers=self.headers,
            timeout=self.timeout,
        ) as session:
            session.headers.setdefault('User-Agent', self.user_agent)
            yield session

    async def download_file(
        self,
        session: aiohttp.ClientSession,
        download_url: str,
        download_path: Path,
    ) -> None:
        response: aiohttp.ClientResponse
        async with session.get(download_url, allow_redirects=False) as response:
            response.raise_for_status()
            ct, _ = cgi.parse_header(response.headers.get('content-type', ''))
            # При кодировании текста вырезаются, т.н. BAD CHARS, что делает невозможным
            # gzip-декодирование git-объектов
            if ct == 'text/html':
                raise ValueError(f"bad content type ({ct}): {download_url}")
            download_path.parent.mkdir(parents=True, exist_ok=True)
            with download_path.open('wb') as fp:
                async for chunk in response.content.iter_chunked(8192):
                    fp.write(chunk)

        logger.info("downloaded: %s", download_url)

    def sha12filename(self, sha1: str) -> str:
        return f'objects/{sha1[:2]}/{sha1[2:]}'

    async def parse(
        self,
        download_path: Path,
        download_url: str,
        download_queue: asyncio.Queue,
    ) -> None:
        git_url = download_url[: download_url.rfind('.git/') + len('.git/')]
        _, filename = str(download_path).rsplit('.git/', 2)
        if filename == 'index':
            # https://git-scm.com/docs/index-format
            hashes = []
            with download_path.open('rb') as fp:
                sig, ver, num_entries = read_struct(fp, '!4s2I')
                assert sig == b'DIRC'
                assert ver in (2, 3, 4)
                assert num_entries > 0
                logger.debug("num entries: %d", num_entries)
                while num_entries:
                    entry_size = fp.tell()
                    fp.seek(40, io.SEEK_CUR)  # file attrs
                    sha1 = fp.read(20).hex()
                    hashes.append(sha1)
                    fp.seek(2, io.SEEK_CUR)  # file flags
                    filename = b''
                    while (c := fp.read(1)) != b'\0':
                        assert c != b''  # Неожиданный конец
                        filename += c
                    logger.debug("%s %s %s", git_url, sha1, filename.decode())
                    entry_size -= fp.tell()
                    # Размер entry кратен 8 (добивается NULL-байтами)
                    fp.seek(entry_size % 8, io.SEEK_CUR)
                    num_entries -= 1
            for sha1 in hashes:
                await download_queue.put(
                    urljoin(git_url, self.sha12filename(sha1))
                )
        elif filename == 'objects/info/packs':
            # Содержит строки вида "P <hex>.pack"
            contents = download_path.read_text()
            for pack in PACK_RE.findall(contents):
                await download_queue.put(
                    urljoin(git_url, f'objects/pack/{pack}.idx')
                )
                await download_queue.put(
                    urljoin(git_url, f'objects/pack/{pack}.pack')
                )
        elif not filename.startswith('objects/'):
            contents = download_path.read_text()
            for match in SHA1_OR_REF_RE.finditer(contents):
                group = match.groupdict()
                if group['sha1']:
                    await download_queue.put(
                        urljoin(git_url, self.sha12filename(group['sha1']))
                    )
                    continue
                for prefix in ['', 'logs/']:
                    await download_queue.put(
                        urljoin(git_url, f"{prefix}{group['ref']}")
                    )
