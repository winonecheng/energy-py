## rewrite notes

new branch rethink

delete all and start again

remove all csv
```
git reflog expire --expire=now --all && git gc --prune=now --aggressive

brew install bfg
bfg -D '*.csv'

while read -r largefile; do
    echo $largefile | awk '{printf "%s %s ", $1, $3 ; system("git rev-list --all --objects | grep " $1 " | cut -d \" \" -f 2-")}'
done <<< "$(git rev-list --all --objects | awk '{print $1}' | git cat-file --batch-check | sort -k3nr | head -n 20)"
```

Went from 50M to 10 M
Writing objects: 100% (6547/6547), 9.68 MiB | 4.20 MiB/s, done.

Added CLI - needed after intstalling binaries from click entry points

```
pyenv rehash
```
