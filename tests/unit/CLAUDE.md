# Unit tests

Pure-logic tests for the `stt_daemon` package: local sockets only, never the mic or the internet.
One folder per package area; shared fixtures and helpers sit at this level.

## Scope

Everything under `tests/unit/`. `audio/`, `backends/`, `session/`, `server/` and `application/` each
have their own CLAUDE.md.

## File tree

Use this tree as a map — scan it to find the parts your task touches, then go straight there.

<!-- BEGIN FILE TREE -->
```
__init__.py     package marker: helpers import as `unit.<folder>.<module>` (pytest puts `tests/` on the path)
conftest.py     fixtures: local fake ElevenLabs server, backend pointed at it, a silent capture
paths.py        repo, fixture and stand-in script paths
waiting.py      wait_until: poll a condition with a timeout
audio/          ring, mic / file / stream / system captures, speakers, VAD, echo cancellers
backends/       backend contract, ElevenLabs realtime and batch on a fake server, Parakeet, TDT, echo gate, plugins
session/        timeline, pause gate, commits, markers, transcripts, recovery, tracks, playback reference, uploads
server/         mic and backend slots (with the slot rig), source scans, speech watch, peer, messages, backend set
application/    CLI, transcribe client, config, credentials, service files
```
<!-- END FILE TREE -->
