# X Accessible Dumper

Dumps everything web accessible: `.git`, `.DS_Store`, sql dumps, backups...

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
$ x-access-dumper -e '\.(png|jpe?g|gif)' -vv https://target 2> err.log
```
