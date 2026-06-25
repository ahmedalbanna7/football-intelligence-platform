from typing import Any


class ReportEngine:
    def build_json_report(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "ok",
            "data": {
                "report_type": "json",
                "sections": context,
            },
            "meta": {
                "engine": "report_stub",
                "future_outputs": ["pdf", "charts", "heatmaps", "highlights"],
            },
        }
