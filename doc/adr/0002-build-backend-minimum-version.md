# ADR 2. Build Backend Minimum Version
## Context
- [Quickstart](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#basic-use) on setuptools doesn't specify a minimum version of the setuptools
- [Python Packaging Authority](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) does mention
- Older build backend version might not support some more modern flags available in toml file

## Option
|     | Option                                         | Pro               | Con                       |
| --- | ----------------------------------------------- | ----------------- | ------------------------- |
| 1   | No minimum bound                               |                   |                           |
| 2   | Use same as PyPackage example                  | Looks more modern |                           |
| 3   | Use elephant pyproject.toml setuptools version | | Very old (v61 vs v84 now) |

## Decision
- Option 2 (i.e. `setuptools >= 77.0`) conflicts with [ADR 3](0003-supported-python-version.md), which keeps
  `requires-python = ">=3.8"` for this patch release: setuptools 77.0 requires Python `>=3.9`
- So for patch release (0.4.1) use Option 3 `setuptools >= 69.0, < 75.4`. Remove upper cap once python 3.8 
support removed
- Pinning setuptools below 77.0 also means giving up SPDX style `license` string +
  top-level `license-files` as mentioned in PEP 639, which only exists from 77.0 onward.
    - Reverted to `license = {file = "LICENSE.txt"}` to match what `setuptools >= 69.0, < 75.4` understands.
	- Change back to how elephant has it once upper pin on setuptools is removed
    - Since the SPDX `license-expression` isn't available under this cap, kept the
      `License :: OSI Approved :: BSD License` classifier as the only machine-readable license
      indicator for now. Metadata 2.4 (setuptools >= 77.0) doesn't allow a classifier alongside
      `license-expression`, so once the setuptools cap is lifted, switch to
      `license = "BSD-3-Clause"` and remove this classifier at the same time.

