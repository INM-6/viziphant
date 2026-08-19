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
