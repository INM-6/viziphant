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
