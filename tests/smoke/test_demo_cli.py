from demo.cli import run_prompt_loop


class FakeService:
    def __init__(self):
        self.prompts = []

    def process_prompt(self, prompt, *, user_id, alias):
        self.prompts.append((prompt, user_id, alias))
        return {"response": "safe answer", "blocked": prompt == "attack", "redacted": False, "layer": "input_guardrail" if prompt == "attack" else None}


def test_prompt_loop_reports_guardrail_result_and_exits_on_quit():
    service = FakeService()
    output = []
    answers = iter(["hello", "attack", "/q"])
    run_prompt_loop(service, input_fn=lambda _: next(answers), output_fn=output.append)
    assert service.prompts == [("hello", "local-operator", "Bạn"), ("attack", "local-operator", "Bạn")]
    assert any("ĐÃ CHẶN" in line for line in output)
    assert any("safe answer" in line for line in output)
    assert any("Kết thúc" in line for line in output)


def test_blank_prompt_is_ignored():
    service = FakeService()
    output = []
    answers = iter(["   ", "/q"])
    run_prompt_loop(service, input_fn=lambda _: next(answers), output_fn=output.append)
    assert service.prompts == []
    assert any("trống" in line.lower() for line in output)
