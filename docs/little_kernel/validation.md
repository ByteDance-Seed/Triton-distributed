# Bundled compiler validation

Source revision: `744899c07d07531eb21262e42bf5bae78715d1e7`.
Bundled hardware: `nv/sm90`, `nv/sm100`, `amd/cdna4`.

The source update uses the selected backend dependency closure and does not
import upstream Git history. Public source checks cover synchronized code,
examples, tests and documentation. New commit messages use public component
names and contain no internal deployment or experiment records.

## Checks performed

| Check | Result |
| --- | --- |
| Entire bundled test tree on CPU (`uv run --locked pytest -q`) | 2803 passed, 95 capability skips, 1209 subtests passed |
| Backend CPU suites (`tools/verify.py --suite cpu`) | `nv/sm90`: 2472 passed; `nv/sm100`: 2265 passed; `amd/cdna4`: 171 passed (0 failed) |
| Parent packaging discovery/resources/build switch (`tests/repository/test_parent_distribution.py`) | 3 tests passed; included above |
| Standalone wheel and isolated installed-wheel backend factories | 3 backends (`amd/cdna4`, `nv/sm90`, `nv/sm100`) passed; module origins checked |
| Backend dependency direction (`tools/check_backends.py`) | 0 violations |
| Repository formatter, lint, license and publication checks | Passed |

The open-source and publication checkers are run with ordinary source exclusions
and with commit-log checking enabled for the final change. No new exclusion or
weakened checker is introduced.

Formatting compatibility changes preserve dynamic DSL imports/exports and
negative tests. Header checks accept equivalent C++ whitespace, and program
rewrites match AST structure rather than a particular Python line wrapping.
No numerical tolerance or kernel algorithm was relaxed to satisfy a test.

## Scope

CPU/mock/source-generation checks do not establish device performance. Full GPU
numerical suites, AMD device execution, Blackwell device execution and
distributed performance require matching hardware, drivers, toolchains and
tensor integrations. Skipped cases are not counted as device passes. The root
locked SDK environment uses Python 3.11 or newer; tensor/benchmark integrations
remain explicit deployment requirements.
