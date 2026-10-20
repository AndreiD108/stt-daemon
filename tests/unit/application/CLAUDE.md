# Application unit tests

Unit tests for the top-level modules and `service/`: CLI (plugins too), `transcribe` client, config, credentials, service files. See `../CLAUDE.md`.

## Scope

Everything under `tests/unit/application/`. No nested CLAUDE.md.

## File tree

Use this tree as a map — scan it to find the parts your task touches, then go straight there.

<!-- BEGIN FILE TREE -->
```
__init__.py                 package marker
test_cli.py                 run (broken config, --keys-stdin), key set and backend install default rule, set-default, backend remove, service files
test_cli_plugins.py         backend list, key set / install / set-default by plugin name
test_config.py              paths per OS, defaults, type errors, Parakeet limits, config writer and key removal
test_credentials.py         mode/owner checks, precedence, writer, stdin parsing, no key in errors
test_service.py             unit/plist rendering, ExecStart escaping, systemctl calls, remove guard, PortAudio, ffmpeg and audiotee hints
test_transcribe_client.py   `transcribe` CLI vs a scripted daemon: file bytes, text and JSON output, exit codes
```
<!-- END FILE TREE -->
