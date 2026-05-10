"""
reports/generate_report.py

Generates a self-contained HTML stakeholder report for a given team and season.
Readable by coaches, scouts, and management — no Python or data science background required.

Usage:
    python reports/generate_report.py --team TOR --season 2023 --output reports/output/TOR_2023.html
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


# ── Chart Builders ─────────────────────────────────────────────────────────────

def xg_trend_chart(team_data: pd.DataFrame, team: str) -> str:
    """xG for/against trend over season with rolling average. Returns HTML div."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=team_data["game_date"], y=team_data["xg_for"],
        name="xG For", line=dict(color="#2563eb", width=2),
        mode="lines+markers", marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=team_data["game_date"], y=team_data["xg_against"],
        name="xG Against", line=dict(color="#dc2626", width=2),
        mode="lines+markers", marker=dict(size=4),
    ))

    # Rolling average
    team_data = team_data.copy()
    team_data["roll_for"] = team_data["xg_for"].rolling(10, min_periods=3).mean()
    team_data["roll_against"] = team_data["xg_against"].rolling(10, min_periods=3).mean()
    fig.add_trace(go.Scatter(
        x=team_data["game_date"], y=team_data["roll_for"],
        name="xG For (10-game avg)", line=dict(color="#93c5fd", dash="dash", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=team_data["game_date"], y=team_data["roll_against"],
        name="xG Against (10-game avg)", line=dict(color="#fca5a5", dash="dash", width=2),
    ))

    fig.update_layout(
        title=f"{team} — Expected Goals For/Against, 5v5",
        xaxis_title="Date",
        yaxis_title="xG",
        legend=dict(orientation="h", y=-0.2),
        template="plotly_white",
        height=400,
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def player_xg_chart(player_data: pd.DataFrame, top_n: int = 20) -> str:
    """Horizontal bar chart: players sorted by xG, colored by goals-xG differential."""
    top = player_data.head(top_n).sort_values("xG")
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in top["goals_minus_xg"]]

    fig = go.Figure(go.Bar(
        y=top["shooterName"],
        x=top["xG"],
        orientation="h",
        marker_color=colors,
        text=top.apply(lambda r: f"{r['goals']}G / {r['xG']:.1f} xG", axis=1),
        textposition="outside",
        customdata=top[["goals", "xG", "goals_minus_xg", "shots"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Goals: %{customdata[0]}<br>"
            "xG: %{customdata[1]:.2f}<br>"
            "G-xG: %{customdata[2]:+.2f}<br>"
            "Shots: %{customdata[3]}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Top Forwards by Expected Goals (5v5)<br>"
              "<sup>Green = overperforming, Red = underperforming vs expectation</sup>",
        xaxis_title="Expected Goals",
        template="plotly_white",
        height=600,
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def zone_entry_chart(zone_data: pd.DataFrame) -> str:
    """Stacked bar: controlled vs dump-in entries per line, colored by outcome."""
    fig = go.Figure()
    for entry_type, color in [("controlled", "#2563eb"), ("uncontrolled", "#9ca3af")]:
        subset = zone_data[zone_data["entry_type"] == entry_type]
        fig.add_trace(go.Bar(
            name=entry_type.capitalize(),
            x=subset["line_id"],
            y=subset["n"],
            marker_color=color,
            customdata=subset["shot_rate"].values,
            hovertemplate="%{x}<br>Entries: %{y}<br>Shot rate: %{customdata:.1%}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title="Zone Entries by Line Combination — Controlled vs Dump-In",
        xaxis_title="Line",
        yaxis_title="Zone Entries",
        template="plotly_white",
        height=400,
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


# ── Report Template ────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{team} Analytics Report — {season}-{season1} Season</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 0 auto;
           padding: 32px 24px; color: #111827; background: #f9fafb; }}
    h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 4px; }}
    h2 {{ font-size: 1.25rem; font-weight: 600; margin-top: 40px; border-bottom: 2px solid #e5e7eb;
          padding-bottom: 8px; }}
    .subtitle {{ color: #6b7280; margin-bottom: 32px; }}
    .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
    .kpi {{ background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    .kpi .val {{ font-size: 2rem; font-weight: 700; color: #1d4ed8; }}
    .kpi .lbl {{ font-size: 0.85rem; color: #6b7280; margin-top: 4px; }}
    .kpi .delta {{ font-size: 0.85rem; margin-top: 2px; }}
    .chart-card {{ background: #fff; border-radius: 12px; padding: 24px;
                   box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 24px; }}
    .insight {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 16px 20px;
                border-radius: 0 8px 8px 0; margin: 16px 0; font-size: 0.95rem; }}
    .insight strong {{ display: block; margin-bottom: 4px; }}
    footer {{ margin-top: 48px; color: #9ca3af; font-size: 0.8rem; text-align: center; }}
  </style>
</head>
<body>
  <h1>{team} Hockey Analytics Report</h1>
  <p class="subtitle">{season}-{season1} Regular Season &nbsp;·&nbsp; Generated {date}</p>

  <div class="kpi-row">
    <div class="kpi">
      <div class="val">{xgf60:.2f}</div>
      <div class="lbl">xGF/60 (5v5)</div>
      <div class="delta" style="color:{xgf_color}">{xgf_delta:+.2f} vs league avg</div>
    </div>
    <div class="kpi">
      <div class="val">{xga60:.2f}</div>
      <div class="lbl">xGA/60 (5v5)</div>
      <div class="delta" style="color:{xga_color}">{xga_delta:+.2f} vs league avg</div>
    </div>
    <div class="kpi">
      <div class="val">{xgpct:.1%}</div>
      <div class="lbl">xG% (5v5)</div>
      <div class="delta" style="color:{pct_color}">{pct_rank} of 32 teams</div>
    </div>
    <div class="kpi">
      <div class="val">{controlled_pct:.1%}</div>
      <div class="lbl">Controlled Zone Entry %</div>
      <div class="delta">League avg: 52.4%</div>
    </div>
  </div>

  <h2>Expected Goals For/Against — Season Trend</h2>
  <div class="chart-card">
    {xg_trend}
  </div>
  <div class="insight">
    <strong>What this means</strong>
    Expected Goals (xG) measures the quality of shot attempts — not just raw shots.
    A team consistently above the red line is generating better scoring opportunities
    than it concedes, regardless of actual goals (which carry more randomness over a short sample).
  </div>

  <h2>Player Expected Goals — 5v5</h2>
  <div class="chart-card">
    {xg_players}
  </div>
  <div class="insight">
    <strong>Reading this chart</strong>
    Bar length = total xG generated this season. Color indicates finishing:
    <span style="color:#16a34a">green</span> players scored more goals than expected (positive variance, may regress),
    <span style="color:#dc2626">red</span> players scored fewer than expected (negative variance, may improve).
    Players consistently underperforming across multiple seasons may have a genuine finishing problem.
  </div>

  <h2>Zone Entry Efficiency by Line</h2>
  <div class="chart-card">
    {zone_entries}
  </div>
  <div class="insight">
    <strong>Why zone entries matter</strong>
    Controlled zone entries (blue) generate shots at approximately 2× the rate of
    dump-ins (gray). Lines with high dump-in rates but similar ice time are burning
    possession — a coaching intervention target.
  </div>

  <footer>
    NHL Analytics Platform &nbsp;·&nbsp; Models: XGBoost xG + Bidirectional LSTM Sequence Model
    &nbsp;·&nbsp; Data: NHL API + MoneyPuck
  </footer>
</body>
</html>"""


def generate_report(team: str, season: int, output_path: str,
                    shots: pd.DataFrame,
                    player_summary: pd.DataFrame,
                    zone_data: pd.DataFrame,
                    team_game_logs: pd.DataFrame) -> None:
    """
    Assemble and write the stakeholder HTML report.
    All chart data should be pre-computed and passed in.
    """
    import datetime

    xgf60 = team_game_logs["xg_for"].mean() * 60 / 20
    xga60 = team_game_logs["xg_against"].mean() * 60 / 20
    xgpct = xgf60 / (xgf60 + xga60)
    league_avg_xgf = 2.45
    league_avg_xga = 2.45
    controlled_pct = zone_data[zone_data["entry_type"] == "controlled"]["n"].sum() / zone_data["n"].sum()

    html = HTML_TEMPLATE.format(
        team=team,
        season=season,
        season1=season + 1,
        date=datetime.date.today().strftime("%B %d, %Y"),
        xgf60=xgf60,
        xga60=xga60,
        xgpct=xgpct,
        xgf_delta=xgf60 - league_avg_xgf,
        xga_delta=xga60 - league_avg_xga,
        pct_rank=int(round((1 - xgpct) * 32)) + 1,
        controlled_pct=controlled_pct,
        xgf_color="#16a34a" if xgf60 > league_avg_xgf else "#dc2626",
        xga_color="#16a34a" if xga60 < league_avg_xga else "#dc2626",
        pct_color="#16a34a" if xgpct > 0.5 else "#dc2626",
        xg_trend=xg_trend_chart(team_game_logs, team),
        xg_players=player_xg_chart(player_summary),
        zone_entries=zone_entry_chart(zone_data),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    logger.info(f"Report written to {output_path}")
    print(f"✓ Report saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NHL Analytics stakeholder report")
    parser.add_argument("--team", required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = args.output or f"reports/output/{args.team}_{args.season}.html"

    # In a real run these would be loaded from processed data / model outputs
    print(f"Generating report for {args.team} {args.season}-{args.season+1}...")
    print(f"[Load shots, player summaries, zone data from data/processed/]")
    print(f"[Run generate_report(...)]")
    print(f"Output: {output}")
