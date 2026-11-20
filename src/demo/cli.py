"""Interactive local prompt tester for the Lab 11 Blue agent."""
from __future__ import annotations


def run_prompt_loop(service, *, input_fn=input, output_fn=print) -> None:
    output_fn("Blue Guardrails · Prompt tester local")
    output_fn("Nhập prompt để thử. Gõ /q hoặc Ctrl+C để thoát. Dùng dữ liệu giả lập, không nhập bí mật thật.\n")
    while True:
        try:
            prompt = input_fn("Prompt> ")
        except (EOFError, KeyboardInterrupt):
            output_fn("\nKết thúc prompt tester.")
            return

        if prompt.strip().casefold() in {"/q", "/quit", "exit", "quit"}:
            output_fn("Kết thúc prompt tester.")
            return
        if not prompt.strip():
            output_fn("Prompt đang trống; nhập lại nhé.\n")
            continue

        try:
            result = service.process_prompt(
                prompt, user_id="local-operator", alias="Bạn"
            )
        except Exception as exc:
            output_fn(f"Lỗi: {exc}\n")
            continue

        if result["blocked"]:
            state = f"ĐÃ CHẶN · {result.get('layer') or 'guardrail'}"
        elif result["redacted"]:
            state = f"ĐÃ CHE THÔNG TIN · {result.get('layer') or 'output_guardrail'}"
        else:
            state = "ĐÃ TRẢ LỜI"
        output_fn(f"[{state}]\nBlue: {result['response']}\n")


def main() -> None:
    from pathlib import Path
    import sys

    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from demo.service import BlueDemoService

    try:
        service = BlueDemoService.from_environment()
    except Exception as exc:
        raise SystemExit(f"Không khởi tạo được Blue: {exc}") from None
    run_prompt_loop(service)


if __name__ == "__main__":
    main()
