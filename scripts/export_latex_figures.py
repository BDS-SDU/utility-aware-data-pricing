from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "blue": (0.0, 0.447, 0.698),
    "orange": (0.902, 0.624, 0.0),
    "green": (0.0, 0.620, 0.451),
    "vermillion": (0.835, 0.369, 0.0),
    "black": (0.13, 0.13, 0.13),
    "gray": (0.40, 0.40, 0.40),
    "light_gray": (0.90, 0.90, 0.90),
}

LABELS = {
    "row_count": "Rows",
    "token_count": "Tokens",
    "dqs_only": "DQS",
    "proxy_gain": "Proxy",
    "unified": "Unified",
    "calibrated_unified": "Calib.",
    "general_instruction": "Instr.",
    "math_reasoning": "Math",
    "code_summarization": "Code",
}


def esc(text: object) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PdfCanvas:
    def __init__(self, width: int = 410, height: int = 235) -> None:
        self.width = width
        self.height = height
        self.commands: List[str] = []

    def color(self, rgb: Sequence[float]) -> None:
        r, g, b = rgb
        self.commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg {r:.3f} {g:.3f} {b:.3f} RG")

    def line_width(self, width: float) -> None:
        self.commands.append(f"{width:.2f} w")

    def rect(self, x: float, y: float, w: float, h: float, rgb: Sequence[float], stroke: bool = True) -> None:
        self.color(rgb)
        self.commands.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re {'B' if stroke else 'f'}")

    def line(self, x1: float, y1: float, x2: float, y2: float, rgb: Sequence[float], width: float = 0.4) -> None:
        self.color(rgb)
        self.line_width(width)
        self.commands.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def text(self, x: float, y: float, text: object, size: int = 8, font: str = "F1", align: str = "left") -> None:
        content = esc(text)
        approx_width = len(str(text)) * size * 0.45
        if align == "center":
            x -= approx_width / 2
        elif align == "right":
            x -= approx_width
        self.commands.append(f"BT /{font} {size} Tf {x:.2f} {y:.2f} Td ({content}) Tj ET")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = "\n".join(self.commands).encode("latin-1", errors="replace")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
            ).encode("ascii"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold >>",
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        ]
        output = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for idx, obj in enumerate(objects, start=1):
            offsets.append(len(output))
            output.extend(f"{idx} 0 obj\n".encode("ascii"))
            output.extend(obj)
            output.extend(b"\nendobj\n")
        xref_offset = len(output)
        output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        output.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        path.write_bytes(output)


def grouped_bar_chart(
    path: Path,
    title: str,
    labels: List[str],
    series: List[Dict[str, object]],
    y_min: float,
    y_max: float,
    y_label: str,
) -> None:
    canvas = PdfCanvas()
    width, height = canvas.width, canvas.height
    left, right, top, bottom = 50, 16, 30, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    plot_bottom = bottom
    canvas.text(left, height - 18, title, size=11, font="F2")
    canvas.text(8, bottom + plot_h / 2, y_label, size=8)
    canvas.rect(left, plot_bottom, plot_w, plot_h, (1, 1, 1), stroke=True)
    for idx in range(6):
        ratio = idx / 5
        value = y_min + (y_max - y_min) * ratio
        y = plot_bottom + plot_h * ratio
        canvas.line(left, y, left + plot_w, y, COLORS["light_gray"], width=0.25)
        canvas.text(left - 4, y - 2, f"{value:.2f}", size=6, align="right")
    group_w = plot_w / max(1, len(labels))
    bar_w = max(3, (group_w - 10) / max(1, len(series)))
    zero_y = plot_bottom + plot_h * ((0 - y_min) / max(1e-12, y_max - y_min))
    for i, label in enumerate(labels):
        center = left + i * group_w + group_w / 2
        canvas.text(center, 22, LABELS.get(label, label), size=7, align="center")
        for j, spec in enumerate(series):
            value = float(spec["values"][i])
            x = left + i * group_w + 5 + j * bar_w
            y_val = plot_bottom + plot_h * ((value - y_min) / max(1e-12, y_max - y_min))
            y = min(zero_y, y_val)
            h = abs(y_val - zero_y)
            canvas.rect(x, y, bar_w - 1, h, spec["color"])  # type: ignore[arg-type]
    legend_x = left
    for idx, spec in enumerate(series):
        x = legend_x + idx * 82
        canvas.rect(x, 7, 7, 7, spec["color"])  # type: ignore[arg-type]
        canvas.text(x + 10, 7, spec["name"], size=7)
    canvas.save(path)


def load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    out = ROOT / "outputs" / "figures"
    exp = load(ROOT / "outputs" / "experiment_report.json")
    val = load(ROOT / "outputs" / "valuation_report.json")
    real_path = ROOT / "outputs" / "real_multidomain_smoke" / "real_multidomain_report.json"
    real = load(real_path) if real_path.exists() else None

    labels = list(exp["ranking_metrics"].keys())
    grouped_bar_chart(
        out / "ranking_metrics.pdf",
        "Ranking Metrics by Estimator",
        labels,
        [
            {"name": "Spearman rho", "values": [exp["ranking_metrics"][x]["spearman_rho"] for x in labels], "color": COLORS["blue"]},
            {"name": "Top-2 overlap", "values": [exp["ranking_metrics"][x]["top2_overlap"] for x in labels], "color": COLORS["orange"]},
        ],
        0.0,
        1.0,
        "Score",
    )

    labels = sorted(exp["actual_gains"].keys())
    grouped_bar_chart(
        out / "source_estimators.pdf",
        "Actual Gain vs Estimated Source Value",
        labels,
        [
            {"name": "Actual", "values": [exp["actual_gains"][x] for x in labels], "color": COLORS["black"]},
            {"name": "Proxy", "values": [exp["estimators_by_source"]["proxy_gain"][x] for x in labels], "color": COLORS["blue"]},
            {"name": "Calib.", "values": [exp["estimators_by_source"]["calibrated_unified"][x] for x in labels], "color": COLORS["vermillion"]},
        ],
        -0.02,
        0.12,
        "Value",
    )

    source_scores = val["source_scores"]
    labels = [x["source_id"] for x in source_scores]
    grouped_bar_chart(
        out / "valuation_components.pdf",
        "Valuation Components by Source",
        labels,
        [
            {"name": "Unified", "values": [x["unified_score"] for x in source_scores], "color": COLORS["blue"]},
            {"name": "DQS", "values": [x["mean_dqs"] for x in source_scores], "color": COLORS["green"]},
            {"name": "Proxy", "values": [x["proxy_gain"] for x in source_scores], "color": COLORS["vermillion"]},
        ],
        0.0,
        0.9,
        "Norm. value",
    )

    if real is not None:
        summary = real["summary"]["mean_ranking_metrics"]
        labels = [x for x in ["row_count", "token_count", "dqs_only", "proxy_gain", "unified", "calibrated_unified"] if x in summary]
        grouped_bar_chart(
            out / "real_multidomain_summary.pdf",
            "Real Multi-Domain Mean Ranking Metrics",
            labels,
            [
                {"name": "Mean rho", "values": [summary[x]["mean_spearman_rho"] for x in labels], "color": COLORS["blue"]},
                {"name": "Top-2", "values": [summary[x]["mean_top2_overlap"] for x in labels], "color": COLORS["orange"]},
            ],
            -0.1,
            1.0,
            "Score",
        )
        domains = real["domain_results"]
        labels = [x["target_domain"] for x in domains]
        grouped_bar_chart(
            out / "real_domain_rho.pdf",
            "Per-Domain Rank Correlation",
            labels,
            [
                {"name": "Rows", "values": [x["ranking_metrics"]["row_count"]["spearman_rho"] for x in domains], "color": COLORS["gray"]},
                {"name": "Tokens", "values": [x["ranking_metrics"]["token_count"]["spearman_rho"] for x in domains], "color": COLORS["orange"]},
                {"name": "Proxy", "values": [x["ranking_metrics"]["proxy_gain"]["spearman_rho"] for x in domains], "color": COLORS["blue"]},
                {"name": "Calib.", "values": [x["ranking_metrics"]["calibrated_unified"]["spearman_rho"] for x in domains], "color": COLORS["vermillion"]},
            ],
            -1.0,
            1.0,
            "Spearman rho",
        )
    print(f"Exported LaTeX PDF figures to {out}")


if __name__ == "__main__":
    main()
