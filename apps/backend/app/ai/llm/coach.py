from typing import Any


class LLMCoach:
    def summarize(self, context: dict[str, Any]) -> dict[str, Any]:
        match_context = context.get("match_context", {})
        events = context.get("events", {}).get("data", {})
        analytics = context.get("analytics", {}).get("data", {})
        development = context.get("development", {}).get("data", {})
        targets = match_context.get("analysis_targets", {})

        primary_players = analytics.get("players_by_team", {}).get("primary_team", [])
        opponent_players = analytics.get("players_by_team", {}).get("opponent_team", [])
        internal_players = analytics.get("players_by_team", {}).get("club_internal", [])
        if not targets.get("include_primary_players", True):
            primary_players = []
        if not targets.get("include_opponent_players", True):
            opponent_players = []

        coach_summary = self._build_summary(
            match_context=match_context,
            events_count=events.get("events_count", 0),
            primary_players_count=len(primary_players),
            opponent_players_count=len(opponent_players),
            internal_players_count=len(internal_players),
        )

        return {
            "status": "ok",
            "data": {
                "coach_summary": coach_summary,
                "player_report": {
                    "primary_team_players": primary_players,
                    "opponent_team_players": opponent_players,
                    "club_internal_players": internal_players,
                    "player_analysis_targets": {
                        "include_primary_players": targets.get(
                            "include_primary_players",
                            True,
                        ),
                        "include_opponent_players": targets.get(
                            "include_opponent_players",
                            True,
                        ),
                    },
                },
                "training_plan": self._build_training_plan(match_context),
                "development_focus": {
                    "recommendations": development.get("recommendations", []),
                    "weaknesses": development.get("weaknesses", []),
                    "drills": development.get("drills", []),
                },
            },
            "meta": {
                "engine": "llm_coach_stub_v2",
                "providers": ["openai", "claude", "gemini", "local_llama"],
                "context_keys": list(context.keys()),
                "mode": "deterministic_summary_before_llm_provider",
            },
        }

    def _build_summary(
        self,
        match_context: dict[str, Any],
        events_count: int,
        primary_players_count: int,
        opponent_players_count: int,
        internal_players_count: int,
    ) -> str:
        match_type = match_context.get("match_type", "unknown")
        analysis_scope = match_context.get("analysis_scope", "unknown")
        primary_team = match_context.get("primary_team_name") or "Primary team"
        opponent_team = match_context.get("opponent_team_name") or "Opponent"

        if match_type in {"internal_scrimmage", "academy_match"}:
            return (
                f"Internal analysis for {primary_team}: "
                f"{internal_players_count} club player tracks and {events_count} "
                "early events were detected. Use this report for player development "
                "and training feedback."
            )

        if analysis_scope == "my_team_full":
            return (
                f"{primary_team} analysis: {primary_players_count} primary-team "
                f"player tracks and {events_count} early events were detected."
            )

        if analysis_scope == "opponent_team_only":
            return (
                f"Opponent team summary for {opponent_team}: player-level detail is "
                f"limited by scope, with {events_count} early events detected."
            )

        if analysis_scope == "opponent_full":
            return (
                f"Opponent analysis for {opponent_team}: {opponent_players_count} "
                f"opponent player tracks and {events_count} early events were detected."
            )

        return (
            f"Full match analysis for {primary_team} vs {opponent_team}: "
            f"{primary_players_count} primary tracks, {opponent_players_count} "
            f"opponent tracks, and {events_count} early events were detected."
        )

    def _build_training_plan(self, match_context: dict[str, Any]) -> dict[str, Any]:
        scope = match_context.get("analysis_scope")
        if scope in {"opponent_full", "opponent_team_only"}:
            return {
                "focus": "opponent_preparation",
                "sessions": [
                    "Review opponent movement and possession candidates.",
                    "Prepare pressing triggers against opponent buildup.",
                ],
            }

        return {
            "focus": "team_development",
            "sessions": [
                "Review player movement and spacing.",
                "Train passing options around possession candidates.",
                "Use detected movement summaries for individual feedback.",
            ],
        }
