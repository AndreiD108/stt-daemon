from pathlib import Path

TESTS = Path(__file__).parents[1]
REPO = TESTS.parent
FIXTURES = TESTS / "fixtures"
TWO_PHRASES = FIXTURES / "two_phrases.wav"
PLUGIN_FIXTURES = FIXTURES / "plugins"
FAKE_PW_RECORD = TESTS / "fake_pw_record.py"
FAKE_PW_CLI = TESTS / "fake_pw_cli.py"
FAKE_AUDIOTEE = TESTS / "fake_audiotee.py"
