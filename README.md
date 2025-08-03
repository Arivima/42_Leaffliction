# 42_Leaffliction
Computer vision project : Image classification by disease recognition on leaves
## Requirements
- python 3.10.13
- uv
## Quickstart
```
uv sync
uv run script.py # or any other script
```
Under the hood, `uv` will:
- create .venv
- install dependencies from uv.lock
- make sure the project uses the pinned python version (.python-version)
## Full set-up
**Install `uv`**
```bash
pipx install uv
```
**Initialize project**
```bash
uv init
```
**Set Python version**
```bash
uv python install 3.10.13
uv python pin 3.10.13
```
Check python path
```bash
which python
/Users/$USER/.pyenv/shims/python
pyenv which python
/Users/$USER/.pyenv/versions/3.10.13/bin/python
python --version
Python 3.10.13
```
**Install dependencies**
```bash
uv add ruff
uv sync
```
**Run a script**
```bash
uv run script.py
```
**Use ruff for linting and format**
- Check linting issues
```bash
uv run ruff check .
```
- Auto-fix linting issues
```bash
uv run ruff check . --fix
```
- Check formatting issues
```bash
uv run ruff format .
```
**Add .gitignore**  
```gitignore
# Virtualenv
.venv/
# Ruff
.ruff_cache/

# Python
__pycache__/
*.pyc

# Datasets
data/

# Lock file should be committed
!uv.lock
```
## TODO
- install and set up pre-commit


