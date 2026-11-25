import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from stream import AntigravityStream  # noqa: E402

_TOOLS = [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}]


def _stream(*deltas: str, response: str = "") -> AntigravityStream:
    events = [{"event": "init", "conversation_id": "c1"}]
    events += [{"event": "step_update", "step_update": {"text_delta": d}} for d in deltas]
    events.append({"event": "result", "result": {"status": "SUCCESS", "response": response or "".join(deltas)}})
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stderr = io.StringIO("")
    proc.poll.return_value = 0
    proc.wait.return_value = 0
    proc.stdout.readline.side_effect = [json.dumps(e) + "\n" for e in events] + [""]
    client = MagicMock()
    client._isolated_gemini_dir = None
    return AntigravityStream(proc=proc, client=client, model="gemini-4-neon", timeout=5.0, tools=_TOOLS)


def _content(stream: AntigravityStream) -> str:
    return "".join(c.choices[0].delta.content or "" for c in stream if c.choices)


class EchoedToolResultTests(unittest.TestCase):
    def test_reply_opening_with_a_tool_result_fails_the_request(self):
        with self.assertRaisesRegex(RuntimeError, "echoed a tool result"):
            _content(_stream("Tool Res", 'ult (call_104):\n{"status": "success"}'))

    def test_leading_whitespace_does_not_hide_the_echo(self):
        with self.assertRaisesRegex(RuntimeError, "echoed a tool result"):
            _content(_stream("\n  Tool Result (call_1):\nok"))

    def test_echo_in_a_response_that_never_streamed_fails_too(self):
        with self.assertRaisesRegex(RuntimeError, "echoed a tool result"):
            _content(_stream(response="Tool Result (call_9):\nok"))

    def test_ordinary_reply_streams_unchanged(self):
        self.assertEqual(_content(_stream("To", "ol installed. Done.")), "Tool installed. Done.")

    def test_short_reply_that_is_a_prefix_is_kept(self):
        self.assertEqual(_content(_stream("Tool")), "Tool")

    def test_quoting_a_tool_result_later_in_a_reply_is_allowed(self):
        text = "Here it is:\nTool Result (call_1): ok"
        self.assertEqual(_content(_stream(text)), text)


if __name__ == "__main__":
    unittest.main()
