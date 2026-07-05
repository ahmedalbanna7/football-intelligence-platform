from typing import Any


class LineupContext:
    def __init__(self, match_context: dict[str, Any]) -> None:
        tactical_identity = match_context.get("tactical_identity", {})
        self.lineup = tactical_identity.get("lineup", []) or []
        self.substitutions = tactical_identity.get("substitutions", []) or []

    def active_players_at(self, minute: int = 0) -> list[dict[str, Any]]:
        active_by_player_id = {
            item.get("player_id"): item
            for item in self.lineup
            if item.get("is_starter", True)
            and int(item.get("start_minute", 0) or 0) <= minute
        }

        for substitution in sorted(
            self.substitutions,
            key=lambda item: (
                int(item.get("minute", 0) or 0),
                int(item.get("second", 0) or 0),
            ),
        ):
            if int(substitution.get("minute", 0) or 0) > minute:
                continue

            player_out_id = substitution.get("player_out_id")
            player_in_id = substitution.get("player_in_id")
            if player_out_id in active_by_player_id:
                active_by_player_id.pop(player_out_id)
            if player_in_id is not None:
                active_by_player_id[player_in_id] = {
                    "player_id": player_in_id,
                    "team_id": substitution.get("team_id"),
                    "jersey_number": substitution.get("player_in_jersey_number"),
                    "starting_zone": substitution.get("player_in_zone"),
                    "expected_zones": substitution.get("expected_zones", []),
                    "is_starter": False,
                    "start_minute": substitution.get("minute", 0),
                    "source": "substitution",
                }

        return [
            item
            for item in active_by_player_id.values()
            if item.get("player_id") is not None
        ]
