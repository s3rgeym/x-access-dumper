__all__ = ('GitDumper',)

import asyncio
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
REF_RE = re.compile(r'\brefs/\S+')
SHA1_OR_REF_RE = re.compile(
    '(?P<sha1>' + SHA1_RE.pattern + ')|(?P<ref>' + REF_RE.pattern + ')'
)


@dataclasses.dataclass
class GitDumper:
    _: dataclasses.KW_ONLY
    headers: LooseHeaders | None = None
    num_workers: int = 50
    output_directory: str | Path = 'output'
    override_existing: bool = False
    timeout: float = 15.0
    user_agent: str = (
        "Mozilla/5.0"
        " (compatible; YandexBot/3.0; +http://yandex.com/bots)"
        " AppleWebKit/537.36 (KHTML, like Gecko)"
        " Chrome/81.0.4044.268"
    )

    def __post_init__(self) -> None:
        if isinstance(self.output_directory, str):
            self.output_directory = Path(self.output_directory)
        if not isinstance(self.timeout, aiohttp.ClientTimeout):
            self.timeout = aiohttp.ClientTimeout(self.timeout)

    def normalize_url(self, url: str) -> str:
        if '://' not in url:
            url = f'http://{url}'
        for x in ('/', '.git/'):
            if not url.endswith(x):
                url += x
        return url

    async def run(self, urls: typing.Sequence[str]) -> None:
        queue = asyncio.Queue()
        normalized_urls = list(map(self.normalize_url, urls))
        for file in COMMON_FILES:
            for git_url in normalized_urls:
                queue.put_nowait(urljoin(git_url, file))

        # Посещенные ссылки
        seen_urls = set()

        # Запускаем задания в фоне
        workers = [
            asyncio.create_task(self.worker(queue, seen_urls))
            for _ in range(self.num_workers)
        ]

        # Ждем пока очередь станет пустой
        await queue.join()

        # Останавливаем задания
        for _ in range(self.num_workers):
            await queue.put(None)

        for w in workers:
            await w

        for git_url in normalized_urls:
            await self.retrieve_source_code(self.url2localpath(git_url))

    async def worker(self, queue: asyncio.Queue, seen_urls: set[str]) -> None:
        async with self.get_session() as session:
            while True:
                try:
                    download_url = await queue.get()

                    if download_url is None:
                        break

                    if download_url in seen_urls:
                        logger.debug("already seen %s", download_url)
                        continue

                    seen_urls.add(download_url)

                    # "https://example.org/Old%20Site/.git/index" -> "output/example.org/Old Site/.git/index"
                    file_path = self.url2localpath(download_url)

                    if self.override_existing or not file_path.exists():
                        try:
                            await self.download_file(
                                session, download_url, file_path
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
                            if file_path.exists():
                                logger.debug("delete: %s", file_path)
                                file_path.unlink()
                            continue
                    else:
                        logger.debug("file exists: %s", file_path)

                    if (pos := download_url.rfind('.git/')) != -1:
                        git_url = download_url[: pos + len('.git/')]
                        await self.parse_git_file(file_path, git_url, queue)
                    elif file_path.name == '.gitignore':
                        await self.parse_gitignore(
                            file_path, download_url, queue
                        )
                except Exception as ex:
                    logger.error("an unexpected error has occurred: %s", ex)
                finally:
                    queue.task_done()

    def url2localpath(self, download_url: str) -> Path:
        return self.output_directory.joinpath(
            unquote(download_url.split('://')[1])
        )

    async def parse_git_file(
        self,
        file_path: Path,
        git_url: str,
        queue: asyncio.Queue,
    ) -> None:
        if file_path.name == 'index':
            # https://git-scm.com/docs/index-format
            hashes = []
            filenames = []
            with file_path.open('rb') as fp:
                sig, ver, num_entries = read_struct(fp, '!4s2I')
                assert sig == b'DIRC'
                assert ver in (2, 3, 4)
                assert num_entries > 0
                logger.debug("num entries: %d", num_entries)
                while num_entries:
                    entry_size = fp.tell()
                    fp.seek(40, io.SEEK_CUR)  # file attrs
                    # 20 байт хеш, 2 байта флаги
                    sha1 = fp.read(22)[:-2].hex()
                    assert len(sha1) == 40
                    hashes.append(sha1)
                    filename = b''
                    while (c := fp.read(1)) != b'\0':
                        assert c != b''  # Неожиданный конец
                        filename += c
                    filename = filename.decode()
                    filenames.append(filename)
                    logger.debug("%s %s", sha1, filename)
                    entry_size -= fp.tell()
                    # Размер entry кратен 8 (добивается NULL-байтами)
                    fp.seek(entry_size % 8, io.SEEK_CUR)
                    num_entries -= 1
            for sha1 in hashes:
                await queue.put(
                    urljoin(git_url, self.get_object_filename(sha1))
                )
            # Пробуем скачать файлы напрямую, если .git не получится
            # восстановить, то, возможно, повезет с db.ini
            for filename in filenames:
                if self.is_web_accessable(filename):
                    # /.git + file = /file
                    await queue.put(urljoin(git_url[:-1], filename))

        elif file_path.name == 'packs':
            # Содержит строки вида "P <hex>.pack"
            contents = file_path.read_text()
            for sha1 in SHA1_RE.findall(contents):
                for ext in ('idx', 'pack'):
                    await queue.put(
                        urljoin(git_url, f'objects/pack/pack-{sha1}.{ext}')
                    )
        elif not re.fullmatch(
            r'(pack-)?[a-f\d]{38}(\.(idx|pack))?', file_path.name
        ):
            for match in SHA1_OR_REF_RE.finditer(file_path.read_text()):
                group = match.groupdict()
                if group['sha1']:
                    await queue.put(
                        urljoin(
                            git_url,
                            self.get_object_filename(group['sha1']),
                        )
                    )
                    continue
                for directory in ('', 'logs/'):
                    await queue.put(
                        urljoin(git_url, f"{directory}{group['ref']}")
                    )

    async def parse_gitignore(
        self, file_path: Path, download_url: str, queue: asyncio.Queue
    ) -> None:
        filenames = []
        with file_path.open('r') as fp:
            for filename in fp:
                filename = filename.split('#')[0].strip()
                if not filename:
                    continue
                if any(c in filename for c in '*[]'):
                    continue
                if filename.startswith('/'):
                    filename = filename[1:]
                filenames.append(filename)
        for filename in filenames:
            if self.is_web_accessable(filename):
                await queue.put(urljoin(download_url, filename))

    async def retrieve_source_code(self, git_path: Path) -> None:
        cmd = (
            f"git --git-dir='{git_path}'"
            f" --work-tree='{git_path.parent}'"
            ' checkout -- .'
        )
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
            logger.error("can't retrieve source: %s", git_path)

    async def download_file(
        self,
        session: aiohttp.ClientSession,
        download_url: str,
        file_path: Path,
    ) -> None:
        response: aiohttp.ClientResponse
        async with session.get(download_url, allow_redirects=False) as response:
            response.raise_for_status()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open('wb') as fp:
                async for chunk in response.content.iter_chunked(8192):
                    fp.write(chunk)
        logger.info("downloaded: %s", download_url)

    @asynccontextmanager
    async def get_session(self) -> typing.AsyncIterable[aiohttp.ClientSession]:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(verify_ssl=False),
            headers=self.headers,
            timeout=self.timeout,
        ) as session:
            session.headers.setdefault('User-Agent', self.user_agent)
            yield session

    def get_object_filename(self, sha1: str) -> str:
        return f'objects/{sha1[:2]}/{sha1[2:]}'

    def is_web_accessable(self, filename: str) -> bool:
        return not filename.lower().endswith(
            ('.php', '.php3', '.php4', '.php5', '.pl', '.jsp')
        )
