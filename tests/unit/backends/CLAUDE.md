# Backends unit tests

Unit tests for `stt_daemon/backends/`: the contract, ElevenLabs realtime and batch against local fakes, Parakeet and its TDT decoder, the echo gate, install and plugins. See `../CLAUDE.md`.

## Scope

Everything under `tests/unit/backends/`. No nested CLAUDE.md.

## File tree

Use this tree as a map — scan it to find the parts your task touches, then go straight there.

<!-- BEGIN FILE TREE -->
```
__init__.py                 package marker
fake_elevenlabs.py          that server: records requests and messages, scripted commit replies
test_backend_base.py        keyterm merge, previous_text trim
test_echo_gate.py           echo frames: delayed / early-stamped echo, near speech over it, per-span match; gate thresholds, joints, word times back
test_elevenlabs_batch.py    upload fields, PCM/WAV body, words and speaker split, HTTP codes, retry, cancel
test_elevenlabs_open.py     open retries and budget, HTTP and message refusals, key kept out of logs
test_elevenlabs_stream.py   URL params, chunking, commits, padding, acks, partials, errors, self-commit
test_fake_backend.py        reply order, empty commits, delays, partials, failures, time limit
test_install.py             unpack once, partial model replaced, bad download / archive leaves nothing
test_parakeet.py            token → word grouping, per-commit decoding, failures, model check
test_parakeet_engine.py     load at open, one recognizer for all, decodes in turn, idle unload, load failure, reload
test_plugins.py             plugin loader and failures, statuses, config and limits, adapter: chunks, s16le, turns, timeout, options; docs example
test_tdt.py                 keyterm BPE split; trie scoring: whole, broken, restarted, open at the end, nested
```
<!-- END FILE TREE -->
