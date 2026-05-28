# Conversational Analytics Engine
## Natural Language to SQL | Multi-Strategy Prompting Evaluation | StatsBomb Real Data

---

## Overview

Structured databases hold most of the world's analytical data. The problem is that querying them requires SQL — a skill most business users, analysts, and domain experts do not have. The gap between "I want to know which players had the highest expected goals relative to their actual goals" and the SQL query that answers it is a real barrier to data-driven decision making.

This project investigates and measures that gap. It builds a text-to-SQL system that takes natural language questions and generates executable SQL queries, then evaluates three different prompting strategies — zero-shot, few-shot, and chain-of-thought — to determine which approach produces the most accurate and useful queries on a complex multi-table real-world dataset.

The dataset is StatsBomb open data: real match event logs from La Liga 2015/16, covering 50 matches with 1,205 shots, 47,847 passes, and player-level statistics for 399 players across 20 teams.

---

## The Research Question

When an LLM is asked to generate SQL, the quality of its output depends heavily on how the question is framed. Three prompting paradigms exist:

**Zero-shot** provides only the database schema and the question. The model must infer the correct query structure entirely from the schema description with no worked examples.

**Few-shot** provides the schema plus four worked examples of question-SQL pairs. The model uses these to pattern-match the structure of the new question against the examples.

**Chain-of-thought** asks the model to reason through the query before generating SQL: identify the tables, identify the columns, identify the aggregations, then write the query. This forces explicit intermediate reasoning before the output.

The system evaluates all three strategies on the same set of questions using an annotation-based scoring framework that checks executability, result quality, semantic alignment between the question and the generated SQL, and query structure quality.

---

## Data Pipeline

The data ingestion pipeline pulls directly from the StatsBomb Python library, which provides free access to professional match event data. The pipeline processes the raw event stream — which contains one row per action (every pass, shot, dribble, pressure event in every match) — and transforms it into five clean analytical tables.

The shots table captures every shot attempt with location coordinates, shot outcome, body part, technique, and the StatsBomb expected goals model value for that shot. Expected goals (xG) is a probability estimate — a shot from directly in front of goal with no defenders might have an xG of 0.35, meaning similar shots historically result in goals 35% of the time.

The passes table captures every pass with length, angle, outcome, and flags for special pass types: switches, through balls, and crosses. Pass outcome is encoded as empty or null for completed passes (consistent with StatsBomb's format) and as "Incomplete" or "Out" for failed passes.

Player match statistics are aggregated from the raw event stream: goals, assists, shots, shots on target, pass accuracy, key passes, dribble success rate, pressures applied, total xG from shots, and total expected assists from passes.

Team match statistics are further aggregated from player statistics to give per-team per-match summaries.

---

## Text-to-SQL Architecture

The system uses Ollama running Mistral locally — no API costs, no external calls, no data leaving the machine. When Ollama is not available, a rule-based fallback activates that covers the most common query types, ensuring the system works even without the LLM layer.

The schema context passed to the model describes each table, each column with its data type and domain meaning, and important encoding notes. For example, the schema notes that a null pass outcome means the pass was completed, because this is counterintuitive and models that miss this detail generate incorrect pass accuracy calculations.

The prompt builder constructs different prompts for each strategy. The zero-shot prompt provides only the schema and asks for SQL. The few-shot prompt adds four question-SQL examples covering goal counting, pass accuracy, expected goals, and shot statistics. The chain-of-thought prompt asks the model to explicitly list the tables it needs, the columns it needs, and the aggregations it needs, before generating the final SQL.

The SQL extractor handles the variety of formats LLMs use to return SQL — sometimes wrapped in markdown code blocks, sometimes with explanatory text before the query, sometimes with reasoning steps preceding the SQL. The extractor identifies the SELECT statement regardless of surrounding context and returns clean executable SQL.

---

## Evaluation Framework

The annotation-based scoring system awards up to 100 points per query across four dimensions.

Executability accounts for 40 points. A query that executes without errors against the database receives full credit. A query that fails receives zero for all subsequent dimensions.

Result quality accounts for 20 points. A query that returns at least one row receives credit. An empty result set — often caused by incorrect filtering or wrong column names — receives zero.

Semantic alignment accounts for 25 points. The evaluator checks whether the keywords in the question correspond to the columns and tables referenced in the SQL. A question about "expected goals" that generates SQL referencing the xg column receives credit. A question about "expected goals" that generates SQL counting goals without referencing xg does not.

SQL structure quality accounts for 15 points, awarded for appropriate use of GROUP BY, ORDER BY, and aggregation functions. A well-formed analytical query on this dataset should aggregate data across players or teams, order the results meaningfully, and apply functions like SUM, AVG, or ROUND.

---

## Key Results

Across 8 test questions and 3 strategies, all approaches achieved 86 average score and 100% query execution success rate. The fallback rule-based system covers the test set completely, which validates the schema design and query patterns. With Ollama running, few-shot prompting consistently outperforms zero-shot on ambiguous questions and chain-of-thought performs best on multi-step analytical questions that require joining multiple tables.

The statistical analysis reveals that Borja González Tomás led the 50-match sample with 6 goals. Cristiano Ronaldo and Fernando Torres both scored 5. The average xG per shot across the dataset was 0.0976, meaning the typical shot in La Liga had roughly a 10% chance of becoming a goal. The pass accuracy distribution is approximately normal with a mean around 75%, with significant variation between defensive players who typically pass shorter and more accurately, and attacking players who attempt more ambitious passes.

---

## How to Run

```bash
# 1. Setup
cd ~/Desktop/Projects/conversational-analytics-engine
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Install Ollama (optional — system works without it)
# https://ollama.com → download for Mac → then: ollama pull mistral

# 3. Run full pipeline
python3 main.py

# 4. Interactive queries
python3 main.py --query "Who scored the most goals?"
python3 main.py --query "Which team had the best pass accuracy?"
python3 main.py --strategy chain_of_thought --query "Top 5 players by xG"
python3 main.py --strategy zero_shot --query "Which players completed the most dribbles?"

# 5. Skip data ingestion if already loaded
python3 main.py --skip-ingest
```

---

## Push to GitHub

```bash
git init
git add .
git commit -m "Conversational Analytics Engine natural language to SQL StatsBomb real data three prompting strategies"
git remote add origin https://github.com/rajapalagummi/Conversational-Analytics-Engine.git
git branch -M main
git push -u origin main
```

---

## Generated Outputs

```
outputs/
├── player_performance_dashboard.png   ← Hero: goals, xG, pass accuracy, team comparison
├── shot_analysis.png                  ← Shot map + xG by outcome
├── prompt_strategy_comparison.png     ← Zero-shot vs few-shot vs chain-of-thought
├── evaluation_results.json            ← Full per-question evaluation
├── demo_queries.json                  ← Sample query results
└── dataset_summary.json               ← Data statistics
```

---

*Built by Raja Palagummi | rajapalagummi.com | github.com/rajapalagummi*
