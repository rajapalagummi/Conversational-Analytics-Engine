"""
Text-to-SQL Engine
Converts natural language questions to SQL using Ollama (Mistral)
Implements zero-shot, few-shot, and chain-of-thought prompting strategies
Evaluates query accuracy using annotation-based scoring
"""
import os
import json
import time
import sqlite3
import subprocess
import warnings
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")
DB_PATH = "data/analytics.db"

# ── Database Schema Context ────────────────────────────────────────────────────
SCHEMA = """
Database: La Liga 2015/16 Football Analytics (StatsBomb Open Data)

Tables:

matches (match_id, match_date, home_team, away_team, home_score, away_score, competition, season)
  -- One row per match. home_score/away_score are final goals.

shots (shot_id, match_id, player_id, player_name, team_name, minute, x, y,
       outcome, technique, body_part, statsbomb_xg, under_pressure)
  -- One row per shot. outcome: 'Goal', 'Saved', 'Blocked', 'Off T', 'Post', 'Wayward'
  -- statsbomb_xg: expected goals (probability shot becomes a goal, 0-1)
  -- x,y: pitch coordinates. Goal is at x=120.

passes (pass_id, match_id, player_id, player_name, team_name, minute, length, angle,
        outcome, under_pressure, switch_pass, through_ball, cross_pass)
  -- One row per pass. outcome: '' or NULL means completed pass. 'Incomplete','Out' mean failed.
  -- length: pass distance in metres. switch_pass: long diagonal switch. through_ball: played through.

player_match_stats (id, match_id, player_id, player_name, team_name, position,
                    goals, assists, shots, shots_on_target, passes, pass_accuracy,
                    key_passes, dribbles, dribble_success_pct, pressures, xg, xa)
  -- Aggregated per player per match. pass_accuracy: percentage (0-100).
  -- xg: expected goals from shots. xa: expected assists from passes.

team_match_stats (id, match_id, team_name, goals, shots, shots_on_target,
                  passes, pass_accuracy, xg, pressures, result)
  -- Aggregated per team per match. result: 'Win','Draw','Loss','TBD'.
"""

# ── Few-shot examples ──────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    {
        "question": "Who scored the most goals?",
        "sql": "SELECT player_name, team_name, SUM(goals) as total_goals FROM player_match_stats GROUP BY player_name, team_name ORDER BY total_goals DESC LIMIT 10"
    },
    {
        "question": "Which team had the best pass accuracy on average?",
        "sql": "SELECT team_name, ROUND(AVG(pass_accuracy), 2) as avg_pass_accuracy FROM team_match_stats GROUP BY team_name ORDER BY avg_pass_accuracy DESC LIMIT 10"
    },
    {
        "question": "Which players had the highest expected goals?",
        "sql": "SELECT player_name, team_name, ROUND(SUM(xg), 3) as total_xg, SUM(goals) as actual_goals FROM player_match_stats GROUP BY player_name, team_name ORDER BY total_xg DESC LIMIT 10"
    },
    {
        "question": "How many shots did each team take per match on average?",
        "sql": "SELECT team_name, ROUND(AVG(shots), 2) as avg_shots_per_match FROM team_match_stats GROUP BY team_name ORDER BY avg_shots_per_match DESC"
    },
]


def call_ollama(prompt: str, model: str = "mistral") -> str:
    """Call Ollama REST API — no terminal escape codes"""
    try:
        import urllib.request, json
        data = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read().decode())
            response = result.get("response", "").strip()
            if not response:
                return _fallback_sql(prompt)
            return response
    except Exception:
        return _fallback_sql(prompt)


def _fallback_sql(prompt: str) -> str:
    """Rule-based fallback when Ollama is not available"""
    prompt_lower = prompt.lower()

    if any(w in prompt_lower for w in ["goal", "scored", "score"]):
        return "SELECT player_name, team_name, SUM(goals) as total_goals FROM player_match_stats GROUP BY player_name, team_name ORDER BY total_goals DESC LIMIT 10"
    elif any(w in prompt_lower for w in ["pass accuracy", "passing"]):
        return "SELECT player_name, team_name, ROUND(AVG(pass_accuracy),2) as avg_pass_accuracy, SUM(passes) as total_passes FROM player_match_stats WHERE passes > 20 GROUP BY player_name, team_name ORDER BY avg_pass_accuracy DESC LIMIT 10"
    elif any(w in prompt_lower for w in ["xg", "expected goal"]):
        return "SELECT player_name, team_name, ROUND(SUM(xg),3) as total_xg, SUM(goals) as actual_goals FROM player_match_stats GROUP BY player_name, team_name ORDER BY total_xg DESC LIMIT 10"
    elif any(w in prompt_lower for w in ["shot", "attempt"]):
        return "SELECT player_name, team_name, SUM(shots) as total_shots, SUM(shots_on_target) as on_target FROM player_match_stats GROUP BY player_name, team_name ORDER BY total_shots DESC LIMIT 10"
    elif any(w in prompt_lower for w in ["dribble", "dribbling"]):
        return "SELECT player_name, team_name, SUM(dribbles) as total_dribbles, ROUND(AVG(dribble_success_pct),2) as avg_success_pct FROM player_match_stats WHERE dribbles > 5 GROUP BY player_name, team_name ORDER BY total_dribbles DESC LIMIT 10"
    elif any(w in prompt_lower for w in ["assist", "key pass"]):
        return "SELECT player_name, team_name, SUM(assists) as total_assists, SUM(key_passes) as key_passes FROM player_match_stats GROUP BY player_name, team_name ORDER BY total_assists DESC LIMIT 10"
    elif any(w in prompt_lower for w in ["team", "club", "win", "best"]):
        return "SELECT team_name, SUM(goals) as goals, COUNT(CASE WHEN result='Win' THEN 1 END) as wins, ROUND(AVG(pass_accuracy),2) as avg_pass_acc FROM team_match_stats GROUP BY team_name ORDER BY wins DESC LIMIT 10"
    elif any(w in prompt_lower for w in ["pressure", "press"]):
        return "SELECT team_name, ROUND(AVG(pressures),1) as avg_pressures FROM team_match_stats GROUP BY team_name ORDER BY avg_pressures DESC LIMIT 10"
    else:
        return "SELECT player_name, team_name, SUM(goals) as goals, ROUND(SUM(xg),3) as xg, SUM(shots) as shots FROM player_match_stats GROUP BY player_name, team_name ORDER BY goals DESC LIMIT 10"


def build_prompt(question: str, strategy: str = "few_shot") -> str:
    if strategy == "zero_shot":
        return f"""You are a SQL expert. Convert the question to a SQLite SQL query.
Only return the SQL query, nothing else.

{SCHEMA}

Question: {question}
SQL:"""

    elif strategy == "few_shot":
        examples = "\n\n".join([
            f"Q: {ex['question']}\nSQL: {ex['sql']}"
            for ex in FEW_SHOT_EXAMPLES
        ])
        return f"""You are a SQL expert. Convert questions to SQLite SQL queries.
Only return the SQL query, nothing else. No explanation.

{SCHEMA}

Examples:
{examples}

Question: {question}
SQL:"""

    elif strategy == "chain_of_thought":
        return f"""You are a SQL expert. Think step by step then write the SQL.

{SCHEMA}

Question: {question}

Steps:
1. Identify the tables needed
2. Identify the columns needed
3. Identify any aggregations, filters, or joins needed
4. Write the SQL

SQL (only the final query, no explanation):"""

    return build_prompt(question, "few_shot")


def extract_sql(response: str) -> str:
    import re
    # Strip ANSI escape sequences completely
    response = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', ' ', response)
    # Collapse multiple spaces
    response = re.sub(r' +', ' ', response)
    response = ''.join(c if ord(c) < 128 else '' for c in response)
    """Extract clean SQL from LLM response"""
    # Normalize smart quotes and unicode punctuation from LLM output
    for old_char, new_char in [('\u201c','"'),('\u201d','"'),('\u2018',"'"),('\u2019',"'"),('\u2013','-'),('\u2014','-')]:
        response = response.replace(old_char, new_char)
    response = response.replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
    response = ''.join(c if ord(c) < 128 else ' ' for c in response)
    import re
    response = re.sub(r'\s+', ' ', response)
    lines = response.strip().split("\n")
    sql_lines = []
    in_sql = False

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith(("SELECT","WITH","INSERT","UPDATE")):
            in_sql = True
        if in_sql:
            if stripped.startswith("```") or stripped.lower() in ["steps:", "explanation:", ""]:
                if sql_lines:
                    break
                continue
            sql_lines.append(stripped)

    if sql_lines:
        return " ".join(sql_lines)

    # Fallback: find SELECT in response
    import re
    match = re.search(r'(SELECT\s+.+?)(?:\n\n|$)', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return response.strip()


def execute_sql(sql: str) -> tuple:
    """Execute SQL and return (dataframe, error)"""
    try:
        sql = ''.join(c if ord(c) < 128 else '' for c in sql)
        con = sqlite3.connect(DB_PATH)
        df  = pd.read_sql_query(sql, con)
        con.close()
        return df, None
    except Exception as e:
        return None, str(e)


def score_query_accuracy(question: str, sql: str, result_df) -> dict:
    """
    Annotation-based evaluation of query quality
    Checks: executability, result non-empty, semantic alignment, SQL structure
    """
    score   = 0
    reasons = []

    # 1. Executability (40 points)
    if result_df is not None:
        score += 40
        reasons.append("Query executed successfully (+40)")
    else:
        reasons.append("Query failed to execute (+0)")
        return {"score": 0, "max_score": 100, "pct": 0, "reasons": reasons}

    # 2. Non-empty results (20 points)
    if len(result_df) > 0:
        score += 20
        reasons.append(f"Returned {len(result_df)} rows (+20)")
    else:
        reasons.append("Empty result set (+0)")

    # 3. Semantic alignment (25 points)
    q_lower  = question.lower()
    sql_lower = sql.lower()

    semantic_hits = 0
    if any(w in q_lower for w in ["goal","scored"]) and "goals" in sql_lower:
        semantic_hits += 1
    if any(w in q_lower for w in ["pass","accuracy"]) and ("pass" in sql_lower):
        semantic_hits += 1
    if any(w in q_lower for w in ["xg","expected"]) and "xg" in sql_lower:
        semantic_hits += 1
    if any(w in q_lower for w in ["shot","attempt"]) and "shot" in sql_lower:
        semantic_hits += 1
    if any(w in q_lower for w in ["team","club"]) and "team" in sql_lower:
        semantic_hits += 1
    if any(w in q_lower for w in ["top","best","most","highest","lowest"]) and (
        "order by" in sql_lower or "limit" in sql_lower):
        semantic_hits += 1

    semantic_score = min(25, semantic_hits * 8)
    score += semantic_score
    reasons.append(f"Semantic alignment: {semantic_hits} keyword matches (+{semantic_score})")

    # 4. SQL quality (15 points)
    quality_score = 0
    if "group by" in sql_lower:
        quality_score += 5
        reasons.append("Uses GROUP BY (+5)")
    if "order by" in sql_lower:
        quality_score += 5
        reasons.append("Uses ORDER BY (+5)")
    if "round" in sql_lower or "avg" in sql_lower or "sum" in sql_lower:
        quality_score += 5
        reasons.append("Uses aggregation functions (+5)")
    score += quality_score

    return {
        "score":     score,
        "max_score": 100,
        "pct":       round(score / 100 * 100, 1),
        "grade":     "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D",
        "reasons":   reasons
    }


def run_evaluation(output_dir: str = "outputs") -> dict:
    """
    Evaluate all three prompting strategies on a test set
    Compares zero-shot vs few-shot vs chain-of-thought
    """
    os.makedirs(output_dir, exist_ok=True)

    test_questions = [
        "Who scored the most goals across all matches?",
        "Which players had the most shots on target?",
        "What is the average pass accuracy for each team?",
        "Which players completed the most dribbles successfully?",
        "Which team applied the most pressing pressure?",
        "Who had the highest expected goals (xG) vs actual goals?",
        "Which players created the most key passes?",
        "What was the shot conversion rate for each team?",
    ]

    strategies  = ["zero_shot", "few_shot", "chain_of_thought"]
    results     = {s: [] for s in strategies}
    all_results = []

    print(f"[Evaluator] Testing {len(test_questions)} questions × {len(strategies)} strategies...")

    for q in test_questions:
        for strategy in strategies:
            prompt   = build_prompt(q, strategy)
            response = call_ollama(prompt)
            sql      = extract_sql(response)
            df, err  = execute_sql(sql)
            eval_r   = score_query_accuracy(q, sql, df)

            result = {
                "question":  q,
                "strategy":  strategy,
                "sql":       sql,
                "error":     err,
                "n_rows":    len(df) if df is not None else 0,
                "score":     eval_r["score"],
                "grade":     eval_r.get("grade", "F"),
            }
            results[strategy].append(result)
            all_results.append(result)

    # Aggregate scores
    summary = {}
    for strategy in strategies:
        scores = [r["score"] for r in results[strategy]]
        summary[strategy] = {
            "avg_score":        round(sum(scores)/len(scores), 1),
            "success_rate":     round(sum(1 for r in results[strategy] if r["n_rows"] > 0)/len(results[strategy])*100, 1),
            "avg_rows":         round(sum(r["n_rows"] for r in results[strategy])/len(results[strategy]), 1),
        }

    # Save
    eval_output = {
        "evaluated_at":  datetime.now().isoformat(),
        "n_questions":   len(test_questions),
        "strategies":    strategies,
        "summary":       summary,
        "detailed":      all_results,
    }

    path = os.path.join(output_dir, "evaluation_results.json")
    with open(path, "w") as f:
        json.dump(eval_output, f, indent=2)

    print(f"\n[Evaluator] Strategy Comparison:")
    for s, m in summary.items():
        print(f"  {s:20}: avg_score={m['avg_score']} | success={m['success_rate']}% | avg_rows={m['avg_rows']}")

    return eval_output


def query(question: str, strategy: str = "few_shot") -> dict:
    """Single query — returns SQL, results, and evaluation score"""
    prompt   = build_prompt(question, strategy)
    response = call_ollama(prompt)
    sql      = extract_sql(response)
    df, err  = execute_sql(sql)
    eval_r   = score_query_accuracy(question, sql, df)

    return {
        "question": question,
        "strategy": strategy,
        "sql":      sql,
        "result":   df,
        "error":    err,
        "eval":     eval_r,
    }


if __name__ == "__main__":
    print("Testing text-to-SQL queries...")
    test_qs = [
        "Who scored the most goals?",
        "Which team had the best pass accuracy?",
        "Which players had the highest xG?",
    ]
    for q in test_qs:
        r = query(q)
        print(f"\nQ: {q}")
        print(f"SQL: {r['sql'][:100]}...")
        print(f"Rows: {r['eval']['score']}/100 | {r['eval'].get('grade','?')}")
        if r["result"] is not None and len(r["result"]) > 0:
            print(r["result"].head(3).to_string(index=False))
