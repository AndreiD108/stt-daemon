# Server unit tests

Unit tests for `stt_daemon/server/`: the mic and backend slots, source scans, daemon-wide VAD, peer lookup, messages and the backend set. See `../CLAUDE.md`.

## Scope

Everything under `tests/unit/server/`. No nested CLAUDE.md.

## File tree

Use this tree as a map — scan it to find the parts your task touches, then go straight there.

<!-- BEGIN FILE TREE -->
```
__init__.py             package marker
slot_harness.py         Slot test rig: recorded events and slot changes, session makers, test clients
test_backend_set.py     setup and default, language default and checks, features, hints, context lookup
test_messages.py        envelope parsing, encoding, per-command field checks
test_peer.py            peer pid/cmdline over real Unix sockets; macOS parsing on fixture bytes
test_slot.py            slot states, sticky evictable, takeover order and failures, client_gone
test_slot_priority.py   dictation priority: victim choice, resume order, backend gone, concurrency refusal retry
test_sources.py         pw-dump parsing on the recording, ids, scan restarts, following an app's next stream, nodes by name
test_sources_macos.py   macOS scan on a stand-in Core Audio: names, ids, listener, polling, read failures, follow by bundle id, scan per OS
test_speech_watch.py    daemon-wide VAD: events, speech starts by id and expiry, mic going quiet
```
<!-- END FILE TREE -->
