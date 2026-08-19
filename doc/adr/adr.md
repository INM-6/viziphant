# ADR 1. Build Backend
## Context

## Options

| Option     | Pro                     | Con                                 |
| ---------- | ----------------------- | ----------------------------------- |
| setuptools | 1. Direct drop-in `setup.py`.<br>2. Supports dynamic fields (version, readme)<br>3. Elephant uses it |                             |
| hatchling  |                         | 1. Needs an extra plugin to do dynamic field for requirement                             |
| uv_build   | 1. "zero config" auto detects stuff<br>2. Clean and concise  | 1. Auto detects stuff, <br>meh better to be explicit<br>2. No dynamic dependency support |

## Decision
Setuptools looks good, maintaining consistency with elephant is given more weightage in this decision

## Status
Proposed
# ADR 2. Build Backend Minimum Version
## Context
- [Quickstart](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#basic-use) on setuptools doesn't specify a minimum version of the setuptools
- [Python Packaging Authority](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) does mention
- Older build backend version might not support some more modern flags available in toml file

## Option
|     | Option                                         | Pro               | Con                       |
| --- | ---------------------------------------------- | ----------------- | ------------------------- |
| 1   | No minimum bound                               |                   |                           |
| 2   | Use same as PyPackage example                  | Looks more modern |                           |
| 3   | Use elephant pyproject.toml setuptools version |                   | Very old (v61 vs v84 now) |
## Decision
Option 2

## Status
Proposed

# ADR 3. Supported Python Version

## Context
- The current minimum supported Python version is `>=3.8`.
release.
- Viziphant, however, is due for a patch release, and a `requires-python` bump feels like a semantic versioning break
-  Python 3.8 is way beyond it's EOL in October 2024, which is a point in

## Decision
Keep the exisiting minimum python version cause this looks like an extreme edge case for people running python 3.8, which is past it's EOL, so can safely just keep it as is for now, and change in next minor release

## Status
Proposed
