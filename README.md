# Virtual Env Setup

So you have auto complete

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirement.txt
```

## How to run

**You only need to run it once.**

```bash
docker build . -t comps
```

For each time you want to use docker

```bash
sh run.sh
```
