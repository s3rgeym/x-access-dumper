__all__ = ('XAccessDumper',)

import asyncio
import cgi
import dataclasses
import io
import itertools
import re
import typing
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import aiohttp
from aiohttp.typedefs import LooseHeaders
from ds_store import DSStore, buddy

from .logger import logger
from .utils import read_struct

OK = 200

GIT_COMMON_FILENAMES = [
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
    '.zshenv',
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

BACKUP_NAMES = ('docroot', 'htdocs', 'www', 'site', 'backup', '{host}')

BACKUP_EXTS = ('.zip', '.tar.gz', '.tgz', '.tar', '.gz')

DB_DUMP_DIRS = (*BACKUP_DIRS, 'sql', 'db', 'database')

DB_DUMP_NAMES = ('dump', 'db', 'database')

DEPLOY_DIRS = ('', 'docker', 'application', 'app', 'api')

DEPLOY_FILENAMES = (
    '.env',
    'prod.env',
    'Dockerfile',
    'docker-compose.yml',
)

CONFIG_DIRS = ('', 'conf', 'config')

CONFIG_NAMES = ('conf', 'config', 'db', 'database')

CONFIG_EXTS = ('.cfg', '.conf', '.ini')

EDIT_FILENAMES = ('index.php', 'wp-config.php', 'conf/db.php')
EDIT_SUFFIXES = ('1', '~', '.bak', '.swp', '.old')

LISTING_DIRS = tuple(
    filter(None, (BACKUP_DIRS + DB_DUMP_DIRS + DEPLOY_DIRS + CONFIG_DIRS))
)


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

    def gen_filenames(self) -> typing.Iterable[str]:
        yield '.git/index'
        yield '.DS_Store'
        for filename in DOT_FILENAMES:
            yield filename
        for dirname in LISTING_DIRS:
            yield dirname + '/'
        for dirname, name, ext in itertools.product(
            BACKUP_DIRS, BACKUP_NAMES, BACKUP_EXTS
        ):
            yield f'{dirname}/{name}{ext}'.lstrip('/')
        for dirname, name in itertools.product(DB_DUMP_DIRS, DB_DUMP_NAMES):
            yield f'{dirname}/{name}.sql'.lstrip('/')
        for dirname, filename in itertools.product(
            DEPLOY_DIRS, DEPLOY_FILENAMES
        ):
            yield f'{dirname}/{filename}'.lstrip('/')
        for dirname, name, ext in itertools.product(
            CONFIG_DIRS, CONFIG_NAMES, CONFIG_EXTS
        ):
            yield f'{dirname}/{name}{ext}'.lstrip('/')
        for filename, suffix in itertools.product(
            EDIT_FILENAMES, EDIT_SUFFIXES
        ):
            yield f'{filename}{suffix}'

    async def run(self, urls: typing.Sequence[str]) -> None:
        queue = asyncio.Queue()
        normalized_urls = list(map(self.normalize_url, urls))
        url_hosts = {x: urlparse(x).netloc for x in normalized_urls}

        # чтобы домены чередовались при запросах
        for filename in self.gen_filenames():
            for url in normalized_urls:
                queue.put_nowait(url + filename.format(host=url_hosts[url]))

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
            git_path = self.url2localpath(url + '.git')
            if git_path.exists():
                await self.retrieve_source_code(git_path)

    async def worker(self, queue: asyncio.Queue, seen_urls: set[str]) -> None:
        async with self.get_session() as session:
            while True:
                try:
                    url = await queue.get()

                    if url is None:
                        break

                    if url in seen_urls:
                        logger.debug("already seen %s", url)
                        continue

                    seen_urls.add(url)

                    if url.endswith('/'):
                        await self.parse_directory_listing(session, url, queue)
                        continue

                    # "https://example.org/Old%20Site/.git/index" -> "output/example.org/Old Site/.git/index"
                    file_path = self.url2localpath(url)

                    if self.exclude_pattern and self.exclude_pattern.search(
                        file_path.name
                    ):
                        logger.debug("exclude file: %s", file_path)
                        continue

                    if self.override_existing or not file_path.exists():
                        try:
                            await self.download_file(session, url, file_path)
                        except Exception as e:
                            match e:
                                case aiohttp.ClientResponseError():
                                    logger.warn(
                                        "%d: %s - %s",
                                        e.status,
                                        e.message,
                                        e.request_info.url,
                                    )
                                case asyncio.exceptions.TimeoutError():
                                    logger.warn("timeout: %s", url)
                                case _:
                                    logger.error(e)
                            if file_path.exists():
                                logger.debug("delete: %s", file_path)
                                file_path.unlink()
                            continue
                    else:
                        logger.debug("file exists: %s", file_path)
                    if file_path.name == '.DS_Store':
                        await self.parse_ds_store(file_path, url, queue)
                    elif (pos := url.rfind('.git/')) != -1:
                        await self.parse_git_file(
                            file_path, url[: pos + len('.git/')], queue
                        )
                    elif file_path.name == '.gitignore':
                        await self.parse_gitignore(file_path, url, queue)
                except Exception as ex:
                    logger.error("an unexpected error has occurred: %s", ex)
                finally:
                    queue.task_done()

    async def parse_directory_listing(
        self, session: aiohttp.ClientSession, url: str, queue: asyncio.Queue
    ) -> None:
        response: aiohttp.ClientResponse
        filenames = []
        try:
            async with session.get(url, allow_redirects=False) as response:
                if response.status != OK:
                    raise ValueError(f"{response.status}: {url}")
                ct, _ = cgi.parse_header(response.headers['content-type'])
                if ct != 'text/html':
                    raise ValueError(f"not text/html: {url}")
                html = await response.text()
                if '<title>Index of /' not in html:
                    raise ValueError(f"not directory listing: {url}")
                for filename in re.findall(r'<a href="([^"]+)', html):
                    # <a href="?C=N;O=D">Name</a>
                    # <a href="/">Parent Directory</a>
                    if not filename.startswith(('/', '?')):
                        filenames.append(filename)
        except (aiohttp.ClientResponseError, aiohttp.ServerTimeoutError):
            logger.warning("request failed: %s", url)
            return
        # закрыли соединение
        for filename in filenames:
            if not self.is_web_accessible(filename):
                continue
            # <a href="backup.zip">
            # <a href="plugins/">
            await queue.put(url + filename)

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
            is_dir = not EXTENSION_RE.search(filename)
            await queue.put(
                urljoin(
                    download_url,
                    # Если файл выглядит как каталог проверяем есть ли в нем
                    # .DS_Store
                    filename + ['', '/.DS_Store'][is_dir],
                )
            )
            # Проверяем листинг
            # if is_dir:
            #     await queue.put(urljoin(download_url, filename + '/'))

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
            for filename in GIT_COMMON_FILENAMES:
                await queue.put(git_url + filename)
            for sha1 in hashes:
                await queue.put(git_url + self.hash2filename(sha1))
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
                            self.hash2filename(group['sha1']),
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
            if response.status != OK:
                raise ValueError(f"{response.status}: {download_url}")
            ct, _ = cgi.parse_header(response.headers['content-type'])
            if ct == 'text/html':
                raise ValueError(f"text/html: {download_url}")
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

    def hash2filename(self, sha1: str) -> str:
        return f'objects/{sha1[:2]}/{sha1[2:]}'

    def is_web_accessible(self, filename: str) -> bool:
        return not filename.lower().endswith(SCRIPT_EXTS)

    def url2localpath(self, download_url: str) -> Path:
        return self.output_directory.joinpath(
            unquote(download_url.split('://')[1])
        )
