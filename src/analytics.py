"""
Analytics Engine
Statistical analysis and visualization of StatsBomb data
Produces all portfolio outputs: player profiles, xG analysis, pass networks, team comparisons
"""
import os
import json
import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")
DB_PATH    = "data/analytics.db"
OUTPUT_DIR = "outputs"

PALETTE = {
    "blue":   "#2E75B6", "red":  "#C93828",
    "green":  "#0A8F5C", "orange": "#B87200",
    "gray":   "#595959", "light": "#E6EAF0"
}


def load(query: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql_query(query, con)
    con.close()
    return df


def generate_player_performance_dashboard():
    """Top performers across all key metrics"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    top_scorers = load("""
        SELECT player_name, team_name, SUM(goals) as goals,
               SUM(shots) as shots, ROUND(SUM(xg),3) as xg,
               SUM(shots_on_target) as sot
        FROM player_match_stats GROUP BY player_name, team_name
        HAVING goals > 0 ORDER BY goals DESC LIMIT 15
    """)

    top_passers = load("""
        SELECT player_name, team_name,
               ROUND(AVG(pass_accuracy),2) as avg_pass_acc,
               SUM(passes) as total_passes, SUM(key_passes) as key_passes
        FROM player_match_stats WHERE passes > 50
        GROUP BY player_name, team_name ORDER BY avg_pass_acc DESC LIMIT 15
    """)

    top_dribblers = load("""
        SELECT player_name, team_name,
               SUM(dribbles) as total_dribbles,
               ROUND(AVG(dribble_success_pct),2) as avg_success
        FROM player_match_stats WHERE dribbles > 5
        GROUP BY player_name, team_name ORDER BY total_dribbles DESC LIMIT 15
    """)

    xg_analysis = load("""
        SELECT player_name, team_name,
               SUM(goals) as goals, ROUND(SUM(xg),3) as xg,
               ROUND(SUM(goals) - SUM(xg), 3) as xg_diff
        FROM player_match_stats WHERE shots > 3
        GROUP BY player_name, team_name ORDER BY xg DESC LIMIT 20
    """)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("La Liga 2015/16 — Player Performance Analytics\nStatsBomb Open Data",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Top Scorers ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if len(top_scorers) > 0:
        y = range(len(top_scorers[:12]))
        bars = ax1.barh([f"{r.player_name[:20]}" for _, r in top_scorers[:12].iterrows()],
                        top_scorers[:12]["goals"],
                        color=PALETTE["blue"], edgecolor="white")
        ax1.set_title("Top Goal Scorers", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Goals"); ax1.spines[["top","right"]].set_visible(False)
        for bar, (_, row) in zip(bars, top_scorers[:12].iterrows()):
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                     f"{int(row.goals)}", va="center", fontsize=8)

    # ── xG vs Actual Goals ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if len(xg_analysis) > 0:
        top_xg = xg_analysis.head(15)
        colors = [PALETTE["green"] if d > 0 else PALETTE["red"] for d in top_xg["xg_diff"]]
        ax2.scatter(top_xg["xg"], top_xg["goals"],
                    c=colors, s=80, alpha=0.8, edgecolors="white", linewidth=0.5)
        max_val = max(top_xg["xg"].max(), top_xg["goals"].max()) + 0.5
        ax2.plot([0, max_val], [0, max_val], "k--", lw=1, alpha=0.4, label="xG = Goals")
        ax2.set_title("Expected Goals (xG) vs Actual Goals", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Expected Goals (xG)"); ax2.set_ylabel("Actual Goals")
        ax2.legend(fontsize=9); ax2.spines[["top","right"]].set_visible(False)

        # Annotate top players
        for _, row in top_xg.head(5).iterrows():
            ax2.annotate(row["player_name"].split()[-1],
                        (row["xg"], row["goals"]),
                        textcoords="offset points", xytext=(5,5), fontsize=7)

    # ── Pass Accuracy Distribution ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    all_pass = load("""
        SELECT player_name, team_name, AVG(pass_accuracy) as avg_acc, SUM(passes) as total
        FROM player_match_stats WHERE passes > 20
        GROUP BY player_name, team_name
    """)
    if len(all_pass) > 0:
        ax3.hist(all_pass["avg_acc"], bins=25, color=PALETTE["blue"],
                 edgecolor="white", alpha=0.85)
        ax3.axvline(all_pass["avg_acc"].mean(), color=PALETTE["red"],
                    linestyle="--", lw=2, label=f"Mean: {all_pass['avg_acc'].mean():.1f}%")
        ax3.set_title("Pass Accuracy Distribution (Players with 20+ passes)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Average Pass Accuracy (%)"); ax3.set_ylabel("Count")
        ax3.legend(fontsize=9); ax3.spines[["top","right"]].set_visible(False)

    # ── Team Comparison ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    team_stats = load("""
        SELECT team_name,
               SUM(goals) as goals, ROUND(AVG(xg),3) as avg_xg,
               ROUND(AVG(pass_accuracy),2) as avg_pass_acc,
               SUM(pressures) as total_pressures
        FROM team_match_stats GROUP BY team_name ORDER BY goals DESC LIMIT 12
    """)
    if len(team_stats) > 0:
        x     = range(len(team_stats))
        width = 0.35
        ax4.bar([i - width/2 for i in x], team_stats["goals"],
                width, label="Goals", color=PALETTE["blue"], alpha=0.85)
        ax4.bar([i + width/2 for i in x], team_stats["avg_xg"] * 10,
                width, label="Avg xG ×10", color=PALETTE["orange"], alpha=0.85)
        ax4.set_title("Team Goals vs Expected Goals", fontsize=12, fontweight="bold")
        ax4.set_xticks(list(x))
        ax4.set_xticklabels([t[:12] for t in team_stats["team_name"]], rotation=35, ha="right", fontsize=8)
        ax4.legend(fontsize=9); ax4.spines[["top","right"]].set_visible(False)

    fig.text(0.5, 0.01,
             f"Data: StatsBomb Open Data | La Liga 2015/16 | Generated: {datetime.now().strftime('%Y-%m-%d')} | rajapalagummi.com",
             ha="center", fontsize=8, color="#999", style="italic")

    path = os.path.join(OUTPUT_DIR, "player_performance_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Analytics] Dashboard → {path}")
    return path


def generate_shot_analysis():
    """Shot quality and xG analysis"""
    shots = load("""
        SELECT s.player_name, s.team_name, s.x, s.y, s.outcome,
               s.statsbomb_xg, s.body_part, s.under_pressure, s.minute
        FROM shots s WHERE s.x IS NOT NULL AND s.y IS NOT NULL
    """)

    if len(shots) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Shot Analysis — La Liga 2015/16", fontsize=14, fontweight="bold")

    # ── Shot Map ──────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#1a5c1a")

    # Pitch markings
    pitch = plt.Rectangle((60, 0), 60, 80, fill=False, color="white", lw=1.5)
    ax1.add_patch(pitch)
    ax1.plot([60, 60], [0, 80], "white", lw=1.5)
    goal = plt.Rectangle((120, 30.34), 0, 19.32, fill=False, color="white", lw=2)
    ax1.add_patch(goal)
    penalty_box = plt.Rectangle((102, 18), 18, 44, fill=False, color="white", lw=1)
    ax1.add_patch(penalty_box)
    six_yard = plt.Rectangle((114, 30), 6, 20, fill=False, color="white", lw=1)
    ax1.add_patch(six_yard)

    goals = shots[shots["outcome"] == "Goal"]
    non_goals = shots[shots["outcome"] != "Goal"]

    ax1.scatter(non_goals["x"], non_goals["y"],
                c=non_goals["statsbomb_xg"], cmap="YlOrRd",
                s=non_goals["statsbomb_xg"]*200 + 20,
                alpha=0.6, edgecolors="white", linewidth=0.3, zorder=3)
    ax1.scatter(goals["x"], goals["y"],
                c="gold", s=100, marker="*",
                edgecolors="white", linewidth=0.5, zorder=5, label="Goal")

    ax1.set_xlim(55, 125); ax1.set_ylim(-5, 85)
    ax1.set_title("Shot Map (size/color = xG value)", fontsize=11, fontweight="bold", color="white")
    ax1.tick_params(colors="white"); ax1.legend(fontsize=9)

    # ── xG by Outcome ─────────────────────────────────────────────
    ax2 = axes[1]
    outcome_xg = shots.groupby("outcome")["statsbomb_xg"].agg(["mean","count"]).reset_index()
    outcome_xg = outcome_xg[outcome_xg["count"] > 5].sort_values("mean", ascending=False)

    colors_bar = [PALETTE["green"] if o == "Goal" else PALETTE["blue"]
                  for o in outcome_xg["outcome"]]
    bars = ax2.bar(outcome_xg["outcome"], outcome_xg["mean"],
                   color=colors_bar, edgecolor="white", alpha=0.85)
    ax2.set_title("Average xG by Shot Outcome", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Shot Outcome"); ax2.set_ylabel("Average xG")
    ax2.set_xticklabels(outcome_xg["outcome"], rotation=25, ha="right")
    for bar, (_, row) in zip(bars, outcome_xg.iterrows()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{row['mean']:.3f}\n(n={int(row['count'])})",
                 ha="center", fontsize=8)
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shot_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Analytics] Shot analysis → {path}")


def generate_prompt_comparison_chart(eval_results: dict):
    """Visual comparison of zero-shot vs few-shot vs chain-of-thought"""
    if not eval_results or "summary" not in eval_results:
        return

    summary = eval_results["summary"]
    strategies = list(summary.keys())
    metrics = ["avg_score", "success_rate"]
    labels  = ["Avg Score (0-100)", "Query Success Rate (%)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Text-to-SQL Prompting Strategy Evaluation\nZero-Shot vs Few-Shot vs Chain-of-Thought",
                 fontsize=13, fontweight="bold")

    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"]]

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        vals = [summary[s][metric] for s in strategies]
        bars = ax.bar(strategies, vals, color=colors, edgecolor="white", alpha=0.85)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(label); ax.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")
        ax.set_xticklabels([s.replace("_", "\n") for s in strategies])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "prompt_strategy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Analytics] Prompt comparison → {path}")


def generate_summary_stats() -> dict:
    stats = {
        "generated_at": datetime.now().isoformat(),
        "data_source":  "StatsBomb Open Data — La Liga 2015/16",
        "matches":      load("SELECT count(*) as n FROM matches").iloc[0]["n"],
        "shots":        load("SELECT count(*) as n FROM shots").iloc[0]["n"],
        "passes":       load("SELECT count(*) as n FROM passes").iloc[0]["n"],
        "players":      load("SELECT count(distinct player_name) as n FROM player_match_stats").iloc[0]["n"],
        "teams":        load("SELECT count(distinct team_name) as n FROM team_match_stats").iloc[0]["n"],
        "top_scorer":   load("SELECT player_name, SUM(goals) as g FROM player_match_stats GROUP BY player_name ORDER BY g DESC LIMIT 1").iloc[0].to_dict() if load("SELECT count(*) as n FROM player_match_stats").iloc[0]["n"] > 0 else {},
        "total_goals":  load("SELECT SUM(goals) as g FROM player_match_stats").iloc[0]["g"],
        "avg_xg_shot":  round(float(load("SELECT AVG(statsbomb_xg) as x FROM shots WHERE statsbomb_xg > 0").iloc[0]["x"]), 4),
    }

    path = os.path.join(OUTPUT_DIR, "dataset_summary.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"[Analytics] Summary → {path}")
    return stats


def run_analytics(eval_results: dict = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[Analytics] Generating analytical outputs...")

    stats = generate_summary_stats()
    generate_player_performance_dashboard()
    generate_shot_analysis()

    if eval_results:
        generate_prompt_comparison_chart(eval_results)

    print(f"\n[Analytics] Dataset: {stats['matches']} matches | {stats['shots']:,} shots | "
          f"{stats['passes']:,} passes | {stats['players']} players")
    if stats.get("top_scorer"):
        print(f"[Analytics] Top scorer: {stats['top_scorer'].get('player_name','')} "
              f"({int(stats['top_scorer'].get('g',0))} goals)")


if __name__ == "__main__":
    run_analytics()
