from __future__ import annotations

from datetime import UTC, datetime
import io
import math
from typing import Any

import cv2
import numpy as np


class ReportsV2Builder:
    def build(
        self,
        analytics: dict[str, Any],
        summary: dict[str, Any],
        team_context: dict[str, Any],
    ) -> dict[str, Any]:
        teams = [analytics["teams"][key] for key in sorted(analytics.get("teams", {}))]
        players = list(analytics.get("players", []))
        return {
            "schema_version": "reports_v2",
            "engine": "canonical_match_reports_v2",
            "generated_at": datetime.now(UTC).isoformat(),
            "analysis_scope": analytics.get("analysis_scope"),
            "quality": {
                "analytics": analytics.get("quality_gate", {}),
                "tracking": (summary.get("tracking_quality") or {}).get("overview", {}),
                "pitch": (summary.get("radar") or {}).get("quality_gate", {}),
                "ball": (summary.get("ball_filter") or {}).get("quality_gate", {}),
                "team_identity": (summary.get("team_classifier") or {}).get("quality_gate", {}),
            },
            "match": {
                "frames": summary.get("frames_processed"),
                "fps": summary.get("fps"),
                "start_frame": summary.get("source_start_frame"),
                "end_frame": summary.get("source_end_frame"),
                "duration_seconds": round(
                    float(summary.get("frames_processed") or 0) / max(float(summary.get("fps") or 25.0), 1e-6),
                    2,
                ),
            },
            "teams": teams,
            "players": players,
            "events": {
                "possession": analytics.get("possession", {}),
                "passing_candidates": analytics.get("passing_candidates", {}),
            },
            "charts": {
                "team_overview": {
                    "metrics": ["total_distance_m", "possession_percent", "space_control_percent"],
                    "series": teams,
                },
                "player_speed": [
                    {
                        "track_id": player.get("track_id"),
                        "team": player.get("team"),
                        "player_name": player.get("player_name"),
                        "average_speed_kmh": player.get("average_speed_kmh"),
                        "max_speed_kmh": player.get("max_speed_kmh"),
                    }
                    for player in players
                ],
            },
            "heatmaps": [
                {
                    "track_id": player.get("track_id"),
                    "team": player.get("team"),
                    "player_name": player.get("player_name"),
                    "heatmap": player.get("heatmap", {}),
                }
                for player in players
                if (player.get("heatmap") or {}).get("samples", 0) > 0
            ],
            "comparison_keys": {
                "team": ["total_distance_m", "average_speed_kmh", "possession_percent", "space_control_percent"],
                "player": ["distance_m", "average_speed_kmh", "max_speed_kmh", "pressure.pressure_index"],
            },
            "rosters": team_context.get("rosters") or {},
        }

    def team_chart_png(self, report: dict[str, Any]) -> bytes:
        canvas = np.full((680, 1180, 3), 248, dtype=np.uint8)
        self._title(canvas, "TEAM COMPARISON", "Quality-gated match metrics")
        teams = report.get("teams", [])
        metrics = [
            ("Distance (m)", "total_distance_m"),
            ("Possession (%)", "possession_percent"),
            ("Space control (%)", "space_control_percent"),
        ]
        colors = [(21, 128, 61), (224, 72, 48)]
        for metric_index, (label, key) in enumerate(metrics):
            top = 145 + metric_index * 165
            cv2.putText(canvas, label, (54, top), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 41, 59), 2, cv2.LINE_AA)
            values = [float(team.get(key) or 0.0) for team in teams[:2]]
            maximum = max(values + [1.0])
            for index, value in enumerate(values):
                y = top + 32 + index * 52
                width = int(820 * value / maximum)
                cv2.rectangle(canvas, (250, y), (250 + width, y + 28), colors[index], cv2.FILLED)
                team_name = self._ascii(teams[index].get("name") or f"Team {index + 1}")
                cv2.putText(canvas, team_name[:20], (54, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 41, 59), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{value:.2f}", (1085, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 41, 59), 1, cv2.LINE_AA)
        return self._encode_png(canvas)

    def heatmap_atlas_png(self, report: dict[str, Any]) -> bytes:
        entries = report.get("heatmaps", [])[:12]
        columns = 3
        rows = max(1, math.ceil(len(entries) / columns))
        tile_width, tile_height = 400, 270
        canvas = np.full((rows * tile_height + 80, columns * tile_width, 3), 245, dtype=np.uint8)
        self._title(canvas, "PLAYER HEATMAPS", "Pitch locations within the analyzed interval")
        for index, entry in enumerate(entries):
            left = (index % columns) * tile_width + 18
            top = (index // columns) * tile_height + 72
            self._draw_pitch_heatmap(canvas, left, top, tile_width - 36, tile_height - 48, entry)
        if not entries:
            cv2.putText(canvas, "Metric heatmaps are blocked until pitch calibration passes.", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (65, 78, 96), 1, cv2.LINE_AA)
        return self._encode_png(canvas)

    def pdf(self, report: dict[str, Any]) -> bytes:
        lines = [
            "FOOTBALL INTELLIGENCE PLATFORM - MATCH REPORT V2",
            f"Generated: {report.get('generated_at', '-')}",
            f"Analysis scope: {report.get('analysis_scope', '-')}",
            f"Frames: {report.get('match', {}).get('frames', '-')} | Duration: {report.get('match', {}).get('duration_seconds', '-')} s",
            "",
            "QUALITY GATES",
        ]
        for key, value in report.get("quality", {}).items():
            lines.append(f"{key}: {value.get('status', 'not available') if isinstance(value, dict) else value}")
        lines.extend(["", "TEAM REPORTS"])
        for team in report.get("teams", []):
            lines.extend([
                f"{self._ascii(team.get('name') or 'Team')}:",
                f"  players={team.get('players_count', 0)} distance_m={team.get('total_distance_m')} avg_speed_kmh={team.get('average_speed_kmh')}",
                f"  possession={team.get('possession_percent')}% space_control={team.get('space_control_percent')}%",
            ])
        lines.extend(["", "PLAYER REPORTS"])
        for player in report.get("players", []):
            label = self._ascii(player.get("player_name") or f"Track {player.get('track_id')}")
            pressure = player.get("pressure") or {}
            lines.append(
                f"{label}: team={player.get('team')} distance_m={player.get('distance_m')} "
                f"avg_speed={player.get('average_speed_kmh')} max_speed={player.get('max_speed_kmh')} "
                f"pressure={pressure.get('pressure_index') if isinstance(pressure, dict) else '-'}"
            )
        return self._simple_pdf(lines)

    def compare(self, reports: list[tuple[int, int, dict[str, Any]]]) -> dict[str, Any]:
        cases = []
        for match_id, run_id, report in reports:
            cases.append({
                "match_id": match_id,
                "run_id": run_id,
                "analysis_scope": report.get("analysis_scope"),
                "teams": report.get("teams", []),
                "players": report.get("players", []),
                "quality": report.get("quality", {}),
            })
        return {
            "schema_version": "reports_v2_comparison",
            "runs": cases,
            "runs_count": len(cases),
            "quality_note": "Only quality-gated metrics are compared; blocked values remain null.",
        }

    def _draw_pitch_heatmap(self, canvas: np.ndarray, left: int, top: int, width: int, height: int, entry: dict[str, Any]) -> None:
        bottom = top + height
        right = left + width
        cv2.rectangle(canvas, (left, top), (right, bottom), (45, 117, 62), cv2.FILLED)
        cv2.rectangle(canvas, (left + 8, top + 8), (right - 8, bottom - 8), (220, 240, 224), 1)
        cv2.line(canvas, ((left + right) // 2, top + 8), ((left + right) // 2, bottom - 8), (220, 240, 224), 1)
        grid = np.array((entry.get("heatmap") or {}).get("grid") or [], dtype=np.float32)
        if grid.size:
            maximum = max(float(np.max(grid)), 1.0)
            cell_height = (height - 16) / grid.shape[0]
            cell_width = (width - 16) / grid.shape[1]
            overlay = canvas.copy()
            for row in range(grid.shape[0]):
                for column in range(grid.shape[1]):
                    intensity = float(grid[row, column]) / maximum
                    if intensity <= 0:
                        continue
                    color = (int(255 * (1.0 - intensity)), int(210 * (1.0 - intensity)), 255)
                    x1 = int(left + 8 + column * cell_width)
                    y1 = int(top + 8 + row * cell_height)
                    x2 = int(left + 8 + (column + 1) * cell_width)
                    y2 = int(top + 8 + (row + 1) * cell_height)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
            cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
        label = self._ascii(entry.get("player_name") or f"Track {entry.get('track_id')}")
        cv2.putText(canvas, label[:28], (left + 8, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (31, 41, 55), 1, cv2.LINE_AA)

    def _title(self, canvas: np.ndarray, title: str, subtitle: str) -> None:
        cv2.putText(canvas, title, (32, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 83, 74), 2, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (32, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (75, 85, 99), 1, cv2.LINE_AA)

    def _encode_png(self, image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Could not encode report chart")
        return encoded.tobytes()

    def _ascii(self, value: Any) -> str:
        text = str(value or "-")
        normalized = text.encode("ascii", "replace").decode("ascii")
        return normalized.replace("?", "_")

    def _simple_pdf(self, lines: list[str]) -> bytes:
        pages = [lines[index:index + 45] for index in range(0, max(len(lines), 1), 45)] or [[]]
        objects: list[bytes] = []
        page_ids = [3 + index * 2 for index in range(len(pages))]
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
        objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
        font_id = 3 + len(pages) * 2
        for page_index, page_lines in enumerate(pages):
            page_id = page_ids[page_index]
            content_id = page_id + 1
            objects.append(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()
            )
            commands = ["BT", "/F1 10 Tf", "42 800 Td", "13 TL"]
            for line in page_lines:
                safe = self._ascii(line).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                commands.append(f"({safe[:105]}) Tj")
                commands.append("T*")
            commands.append("ET")
            stream = "\n".join(commands).encode("latin-1", "replace")
            objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

        output = io.BytesIO()
        output.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for object_id, content in enumerate(objects, 1):
            offsets.append(output.tell())
            output.write(f"{object_id} 0 obj\n".encode())
            output.write(content)
            output.write(b"\nendobj\n")
        xref = output.tell()
        output.write(f"xref\n0 {len(objects) + 1}\n".encode())
        output.write(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.write(f"{offset:010d} 00000 n \n".encode())
        output.write(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
        return output.getvalue()
