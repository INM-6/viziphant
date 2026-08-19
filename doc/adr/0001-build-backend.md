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
