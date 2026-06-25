from typing import Any


class LLMCoach:
    def summarize(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "ok",
            "data": {
                "coach_summary": "LLM coach is ready for future provider integration.",
                "player_report": {},
                "training_plan": {},
            },
            "meta": {
                "engine": "llm_coach_stub",
                "providers": ["openai", "claude", "gemini", "local_llama"],
                "context_keys": list(context.keys()),
            },
        }
