# stt-daemon

A local speech-to-text daemon in Python: one process owns the mic, many clients connect over a Unix
socket, and one session runs at a time. The wire protocol is in `docs/protocol.md`; code against it.

## Scope

Overview of the whole repository. No nested CLAUDE.md.

## File tree

Use this tree as a map — scan it to find the parts your task touches, then go straight there.

<!-- BEGIN FILE TREE -->
```
.github/                  CI
  workflows/
    ci.yml                ruff, pyright, pytest (unit + integration) on push and PR
docs/                     agent- and client-facing docs
  protocol.md             wire protocol: every command and event
src/                      package source
  stt_daemon/             the `stt-daemon` package
    __init__.py           __version__ from package metadata
    cli.py                `stt-daemon` entry point; only --version so far
    config.py             config.toml loading, defaults, validation; config/socket paths per OS; in-place key writer
    credentials.py        API keys: mode-600 credentials.toml, <BACKEND>_API_KEY env, stdin keys; key writer
tests/
  conftest.py             marks everything under tests/live as `live` (skipped by default)
  unit/                   pure logic, no mic, no network
    test_config.py        paths per OS, defaults, type errors, config writer
    test_credentials.py   mode/owner checks, precedence, writer, stdin parsing, no key in errors
  integration/            real daemon over a socket, WAV capture, fake backend
  live/                   opt-in (`pytest -m live`): real mic, real backend key
pyproject.toml            package, deps, ruff/pyright/pytest settings
```
<!-- END FILE TREE -->

## Discipline

- Source code files cap at ~500 lines (tests included). Split by concern if crossing.
- Above rule applies even to files you have not authored yourself. If you touch it, you fix it.
- A directory's listing — its files and its subdirectories with their one-liners — reads as the complete picture of one module at that level.
- CLAUDE.md file trees mirror reality. Update the relevant CLAUDE.md in the same change as a file add/remove/rename/repurpose.
- Tree descriptions: few words, elliptical, what the file is or a rough overview of its contents — not a sentence; ≤120 chars.
- Directories with their own CLAUDE.md get a folder summary, not a recursive tree.

## Checks

`uv run ruff check`, `uv run ruff format --check`, `uv run pyright` (strict), `uv run pytest`. CI runs the
same. Dependency resolution is limited to releases at least a week old (`[tool.uv] exclude-newer`).

## API keys never reach output

No error message, log line or repr may hold a key. Errors about keys name the file, line or variable,
never the value; tests assert this.

## sherpa-onnx needs sherpa-onnx-core pinned beside it

The Linux wheels of `sherpa-onnx` load `libonnxruntime.so` from `sherpa-onnx-core`, but uv resolves
`sherpa-onnx`'s deps from wheels that don't declare it. Keep both as direct deps at the same `==`
version, and bump them together.
