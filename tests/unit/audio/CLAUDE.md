# Audio unit tests

Unit tests for `stt_daemon/audio/`: the ring, mic, file, stream and system captures, speakers, VAD and echo cancelling. See `../CLAUDE.md` for the shared helpers.

## Scope

Everything under `tests/unit/audio/`. No nested CLAUDE.md.

## File tree

Use this tree as a map — scan it to find the parts your task touches, then go straight there.

<!-- BEGIN FILE TREE -->
```
__init__.py                package marker
test_aec.py                built-in canceller on synthetic echo: echo drop, near end kept, delay search and changes, bypass, CPU; Parakeet (slow)
test_aec_builtin.py        built-in canceller as mic track: late-stamped far frames, far gap and restart, mic jump, settle at stop
aec_stand_ins.py           stand-ins for the echo-cancel tests: node table, pushed and flaky mics, pw-cli / pw-record command lines
test_aec_mic.py            shared mic on the module: load at start or per open, fallback, reload, no time going back, watchers, settle
test_aec_pipewire.py       echo-cancel module via a pw-cli stand-in: command, node wait, load errors, kill; mode; install check
test_audiotee_capture.py   audiotee with a stand-in: command, frames, permission hint, exit error, quiet process kept, new pid followed
test_capture.py            device choice, holds, callback, stall/status recovery, backoff, hung stop
test_frames.py             ring order, since-trimming, eviction, multi-attach, attach atomicity
test_pw_capture.py         system capture with a stand-in: frames, lost/restored, backoff, stall, release, following an app
test_speakers.py           SpeakerMonitor: hold during the open, held capture outlives want(False), per-session opens, missing tool
test_stream_capture.py     byte stream with a stand-in decoder: order, audio-time stamps, failures, backpressure
test_vad.py                edges on the fixture vs chunk size and window phase, delays, ids, gap reset
test_wav_capture.py        WAV reading, real-time pacing, per-session open
```
<!-- END FILE TREE -->
