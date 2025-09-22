# 42_Leaffliction
Computer vision project : Image classification by disease recognition on leaves

## Requirements
- python 3.10.13
- uv
## Quickstart
### Install project
This project uses uv to handle dependencies and virtual env
```bash
uv sync # equivalent to pip install .
uv run scripts/my_script.py
```
Under the hood, `uv` will:
- create .venv
- install dependencies from uv.lock
- make sure the project uses the pinned python version (.python-version)
### Project structure
```bash
➜  42_Leaffliction git:(main) tree -L 2  
.
├── en.subject.pdf
├── images # original data : can be named whatever but needs to have this structure
│   ├── Apple
│   └── Grape
├── images_augmented # augmented data : this needs to be copied from original data as the Augmentation.py will modify in-place
│   ├── Apple
│   └── Grape
├── LICENSE
├── plots # this will be created by Distribution.py
│   ├── Apple_combined_chart.png
│   └── Grape_combined_chart.png
├── pyproject.toml
├── README.md
├── scripts
│   ├── Augmentation.py
│   ├── Distribution.py
│   ├── Transformation.py
│   └── utils
└── uv.lock
```

## Full set-up flow used to create this project from scratch
#### **Install `uv`**
```bash
pipx install uv
```
#### **Initialize project**
```bash
uv init
```
#### **Set Python version**
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
#### **Install dependencies**
```bash
uv add ruff # for example
uv sync
```
#### **Run a script**
```bash
uv run script.py
```
#### **Use ruff for linting and format**
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
Can also be done through vscode shortcut : `Cmd` + `Shift` + `P`
- Ruff : Format document
- Ruff : Format imports
- Ruff : Fix all auto-fixable problems  

#### **Add .gitignore**  
```gitignore
# Virtualenv
.venv/
# Ruff
.ruff_cache/

# Python
__pycache__/
*.pyc

# Datasets
images/
images_augmented/
plots/

# Lock file should be committed
!uv.lock
```

