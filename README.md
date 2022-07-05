# X-Access-Dumper

Dumps everything web accessible: git repos, files from `.DS_Store`, sql dumps, backups, configs...

Use asdf or pyenv to install latest python version.

Install:

```bash
$ pip install x-access-dumper
$ pipx install x-access-dumper
```

Usage:

```
$ x-access-dumper -h
$ x-access-dumper url1 url2 url3
$ x-access-dumper < urls.txt
$ command | x-access-dumper
$ x-access-dumper -vv https://target 2> log.txt
```

# TODO:

- <s>exclude images and media files by default</s> 
