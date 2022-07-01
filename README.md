# Web Dumper

Web Dumper Tool.

Use asdf or pyenv to install latest python version.

Features:

- Mass Dumping.
- Dump Git repo.
- Parse `.gitignore`.
- Parse `.DS_Store`.
- Can download web accessible files (eg sql dumps and backups).

Install:

```bash
$ pip install webdumper
$ pipx install webdumper
```

Usage:

```
$ webdumper -h
$ webdumper url1 url2 url3
$ webdumper < urls.txt
$ command | webdumper
$ webdumper url 2> err.log
```
