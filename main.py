"""
Conversational Analytics Engine — Main Orchestrator
Natural Language to SQL | Multi-Strategy Prompting | StatsBomb Real Data

Usage:
    python3 main.py                  # Full pipeline
    python3 main.py --query "Who scored the most goals?"
    python3 main.py --strategy chain_of_thought --query "Which team pressed the most?"
    python3 main.py --evaluate       # Run full prompting strategy evaluation
    python3 main.py --skip-ingest    # Skip data loading (use existing DB)
"""
import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline import run_pipeline
from src.text_to_sql   import query, run_evaluation
from src.analytics     import run_analytics

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║      Conversational Analytics Engine                             ║
║      Natural Language to SQL | StatsBomb La Liga 2015/16         ║
║                                                                  ║
║  Stage 1: StatsBomb Data Ingestion (50 real matches)             ║
║  Stage 2: Text-to-SQL Evaluation (3 prompting strategies)        ║
║  Stage 3: Statistical Analysis & Visualizations                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

DEMO_QUESTIONS = [
    "Who scored the most goals?",
    "Which players had the highest expected goals (xG)?",
    "What is the average pass accuracy for each team?",
    "Which players completed the most dribbles?",
    "Which team applied the most pressing pressure?",
    "Who had the most assists and key passes?",
]


def run_full_pipeline(skip_ingest: bool = False):
    print(BANNER)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data",    exist_ok=True)

    # Stage 1
    if not skip_ingest:
        print("=" * 60)
        print("  STAGE 1: Data Ingestion — StatsBomb La Liga 2015/16")
        print("=" * 60)
        run_pipeline()
    else:
        print("[Pipeline] Skipping ingestion — using existing database")

    # Stage 2: Evaluation
    print("\n" + "=" * 60)
    print("  STAGE 2: Text-to-SQL Prompting Strategy Evaluation")
    print("=" * 60)
    eval_results = run_evaluation("outputs")

    # Stage 3: Analytics
    print("\n" + "=" * 60)
    print("  STAGE 3: Statistical Analysis & Visualizations")
    print("=" * 60)
    run_analytics(eval_results)

    # Demo queries
    print("\n" + "=" * 60)
    print("  DEMO: Sample Natural Language Queries")
    print("=" * 60)
    demo_results = []
    for q in DEMO_QUESTIONS:
        r = query(q, strategy="few_shot")
        print(f"\nQ: {q}")
        print(f"SQL: {r['sql'][:120]}...")
        if r["result"] is not None and len(r["result"]) > 0:
            print(r["result"].head(3).to_string(index=False))
        print(f"Score: {r['eval']['score']}/100")
        demo_results.append({
            "question": q,
            "sql":      r["sql"],
            "score":    r["eval"]["score"],
            "n_rows":   len(r["result"]) if r["result"] is not None else 0,
        })

    # Save demo results
    with open("outputs/demo_queries.json", "w") as f:
        json.dump(demo_results, f, indent=2)

    print(f"""
{'='*60}
  ✓ PIPELINE COMPLETE
{'='*60}

📁 Outputs:
   outputs/player_performance_dashboard.png  ← Hero image
   outputs/shot_analysis.png                 ← xG + shot map
   outputs/prompt_strategy_comparison.png    ← Evaluation chart
   outputs/evaluation_results.json           ← Full evaluation
   outputs/demo_queries.json                 ← Sample queries
   outputs/dataset_summary.json              ← Data statistics

💬 Try your own queries:
   python3 main.py --query "Who scored the most goals?"
   python3 main.py --query "Which team had the best pass accuracy?"
   python3 main.py --strategy chain_of_thought --query "Top 5 players by xG"

📊 Prompting strategies: zero_shot | few_shot | chain_of_thought
""")


def single_query_mode(question: str, strategy: str):
    """Interactive single query"""
    print(f"\nStrategy: {strategy}")
    print(f"Question: {question}\n")

    r = query(question, strategy=strategy)

    print(f"Generated SQL:")
    print(f"  {r['sql']}\n")

    if r["error"]:
        print(f"Error: {r['error']}")
    elif r["result"] is not None:
        print(f"Results ({len(r['result'])} rows):")
        print(r["result"].to_string(index=False))

    print(f"\nEvaluation: {r['eval']['score']}/100 (Grade: {r['eval'].get('grade','?')})")
    for reason in r["eval"]["reasons"]:
        print(f"  {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversational Analytics Engine")
    parser.add_argument("--query",       type=str, default=None)
    parser.add_argument("--strategy",    type=str, default="few_shot",
                        choices=["zero_shot","few_shot","chain_of_thought"])
    parser.add_argument("--evaluate",    action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    if args.query:
        if not os.path.exists("data/analytics.db"):
            print("[Setup] Database not found. Running ingestion first...")
            run_pipeline()
        single_query_mode(args.query, args.strategy)
    elif args.evaluate:
        if not os.path.exists("data/analytics.db"):
            run_pipeline()
        eval_results = run_evaluation("outputs")
        run_analytics(eval_results)
    else:
        run_full_pipeline(skip_ingest=args.skip_ingest)
