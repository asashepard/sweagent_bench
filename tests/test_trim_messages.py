"""Smoke tests for sweagent_runner._trim_messages."""
from sweagent_bench.generation.sweagent_runner import _trim_messages


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def test_trim_keeps_system_and_task_and_last_turns() -> None:
    messages = [
        _msg("system", "sys"),
        _msg("user", "task"),
    ]
    # Add 12 turns (24 messages)
    for i in range(12):
        messages.append(_msg("assistant", f"a{i}"))
        messages.append(_msg("user", f"u{i}"))

    trimmed = _trim_messages(messages, max_turns=8)

    # Must keep system + task
    assert trimmed[0]["role"] == "system" and trimmed[0]["content"] == "sys"
    assert trimmed[1]["role"] == "user" and trimmed[1]["content"] == "task"

    # Remaining should be at most 16 messages (= 8 turns)
    assert len(trimmed) <= 18

    # Ensure tail was preserved
    tail_contents = [m["content"] for m in trimmed]
    assert "a11" in tail_contents and "u11" in tail_contents


def test_trim_without_system() -> None:
    messages = [_msg("user", "task")]
    for i in range(5):
        messages.append(_msg("assistant", f"a{i}"))
        messages.append(_msg("user", f"u{i}"))

    trimmed = _trim_messages(messages, max_turns=2)
    assert trimmed[0]["role"] == "user" and trimmed[0]["content"] == "task"
    # task + 4 tail messages
    assert len(trimmed) <= 5


if __name__ == "__main__":
    test_trim_keeps_system_and_task_and_last_turns()
    test_trim_without_system()
    print("test_trim_messages: OK")
