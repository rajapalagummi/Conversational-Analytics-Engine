"""
Data Pipeline — StatsBomb Open Data Ingestion
Loads real La Liga 2015/16 match data into SQLite
Uses flat column format from statsbombpy
"""
import os
import sqlite3
import warnings
import numpy as np
import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")

DB_PATH     = "data/analytics.db"
COMPETITION = 11
SEASON      = 27
MAX_MATCHES = 50


def setup_db():
    os.makedirs("data", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        DROP TABLE IF EXISTS matches;
        DROP TABLE IF EXISTS shots;
        DROP TABLE IF EXISTS passes;
        DROP TABLE IF EXISTS player_match_stats;
        DROP TABLE IF EXISTS team_match_stats;

        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY, match_date TEXT,
            home_team TEXT, away_team TEXT,
            home_score INTEGER, away_score INTEGER,
            competition TEXT, season TEXT
        );

        CREATE TABLE shots (
            shot_id TEXT PRIMARY KEY, match_id INTEGER,
            player_id INTEGER, player_name TEXT, team_name TEXT,
            minute INTEGER, x REAL, y REAL,
            outcome TEXT, technique TEXT, body_part TEXT,
            statsbomb_xg REAL, under_pressure INTEGER
        );

        CREATE TABLE passes (
            pass_id TEXT PRIMARY KEY, match_id INTEGER,
            player_id INTEGER, player_name TEXT, team_name TEXT,
            minute INTEGER, length REAL, angle REAL,
            outcome TEXT, under_pressure INTEGER,
            switch_pass INTEGER, through_ball INTEGER, cross_pass INTEGER
        );

        CREATE TABLE player_match_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER, player_id INTEGER,
            player_name TEXT, team_name TEXT, position TEXT,
            goals INTEGER, assists INTEGER, shots INTEGER,
            shots_on_target INTEGER, passes INTEGER,
            pass_accuracy REAL, key_passes INTEGER,
            dribbles INTEGER, dribble_success_pct REAL,
            pressures INTEGER, xg REAL, xa REAL
        );

        CREATE TABLE team_match_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER, team_name TEXT,
            goals INTEGER, shots INTEGER, shots_on_target INTEGER,
            passes INTEGER, pass_accuracy REAL,
            xg REAL, pressures INTEGER, result TEXT
        );
    """)
    con.commit()
    con.close()


def process_match(con, match_id, home_team, away_team, home_score, away_score, match_date):
    try:
        ev = sb.events(match_id=match_id)
    except Exception as e:
        return

    mid = int(match_id)

    # ── Shots ────────────────────────────────────────────────────
    shots = ev[ev["type"] == "Shot"].copy()
    for _, s in shots.iterrows():
        loc = s.get("location") or [None, None]
        con.execute("INSERT OR IGNORE INTO shots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            str(s["id"]), mid,
            int(s["player_id"]) if pd.notna(s.get("player_id")) else 0,
            str(s.get("player", "")),
            str(s.get("team", "")),
            int(s.get("minute", 0)),
            float(loc[0]) if loc and loc[0] else None,
            float(loc[1]) if loc and len(loc) > 1 and loc[1] else None,
            str(s.get("shot_outcome", "")),
            str(s.get("shot_technique", "")),
            str(s.get("shot_body_part", "")),
            float(s.get("shot_statsbomb_xg", 0) or 0),
            int(bool(s.get("under_pressure", False))),
        ))

    # ── Passes ───────────────────────────────────────────────────
    passes = ev[ev["type"] == "Pass"].copy()
    for _, p in passes.iterrows():
        outcome = str(p.get("pass_outcome", "")) if pd.notna(p.get("pass_outcome")) else "Complete"
        con.execute("INSERT OR IGNORE INTO passes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            str(p["id"]), mid,
            int(p["player_id"]) if pd.notna(p.get("player_id")) else 0,
            str(p.get("player", "")),
            str(p.get("team", "")),
            int(p.get("minute", 0)),
            float(p.get("pass_length", 0) or 0),
            float(p.get("pass_angle", 0) or 0),
            outcome,
            int(bool(p.get("under_pressure", False))),
            int(bool(p.get("pass_switch", False))),
            int(bool(p.get("pass_through_ball", False))),
            int(bool(p.get("pass_cross", False))),
        ))

    # ── Player Stats ─────────────────────────────────────────────
    ps = {}
    for _, e in ev.iterrows():
        pid  = int(e["player_id"]) if pd.notna(e.get("player_id")) else 0
        if pid == 0: continue
        name = str(e.get("player", ""))
        team = str(e.get("team", ""))
        pos  = str(e.get("position", ""))
        etype = str(e.get("type", ""))

        if pid not in ps:
            ps[pid] = {"name": name, "team": team, "pos": pos,
                       "goals":0,"assists":0,"shots":0,"sot":0,
                       "passes":0,"pass_complete":0,"kp":0,
                       "dribbles":0,"drib_success":0,"pressures":0,
                       "xg":0.0,"xa":0.0}

        if etype == "Shot":
            ps[pid]["shots"] += 1
            ps[pid]["xg"] += float(e.get("shot_statsbomb_xg", 0) or 0)
            outcome = str(e.get("shot_outcome", ""))
            if outcome == "Goal": ps[pid]["goals"] += 1
            if outcome in ["Goal","Saved","Saved To Post"]: ps[pid]["sot"] += 1

        elif etype == "Pass":
            ps[pid]["passes"] += 1
            outcome = str(e.get("pass_outcome", ""))
            if pd.isna(e.get("pass_outcome")) or outcome == "": ps[pid]["pass_complete"] += 1
            if bool(e.get("pass_goal_assist", False)): ps[pid]["assists"] += 1
            if bool(e.get("pass_shot_assist", False)): ps[pid]["kp"] += 1

        elif etype == "Dribble":
            ps[pid]["dribbles"] += 1
            if str(e.get("dribble_outcome", "")) == "Complete": ps[pid]["drib_success"] += 1

        elif etype == "Pressure":
            ps[pid]["pressures"] += 1

    for pid, s in ps.items():
        pa = round(s["pass_complete"]/s["passes"]*100,2) if s["passes"] > 0 else 0
        da = round(s["drib_success"]/s["dribbles"]*100,2) if s["dribbles"] > 0 else 0
        con.execute("""INSERT INTO player_match_stats
            (match_id,player_id,player_name,team_name,position,
             goals,assists,shots,shots_on_target,passes,pass_accuracy,
             key_passes,dribbles,dribble_success_pct,pressures,xg,xa)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            mid,pid,s["name"],s["team"],s["pos"],
            s["goals"],s["assists"],s["shots"],s["sot"],
            s["passes"],pa,s["kp"],s["dribbles"],da,
            s["pressures"],round(s["xg"],4),round(s["xa"],4)
        ))

    con.commit()


def run_pipeline():
    print("""
╔══════════════════════════════════════════════════════╗
║  Data Pipeline — StatsBomb La Liga 2015/16           ║
║  Real match data: 50 matches, shots, passes, stats   ║
╚══════════════════════════════════════════════════════╝
""")
    setup_db()
    matches = sb.matches(competition_id=COMPETITION, season_id=SEASON).head(MAX_MATCHES)

    con = sqlite3.connect(DB_PATH)
    # Load matches
    for _, m in matches.iterrows():
        con.execute("INSERT OR IGNORE INTO matches VALUES (?,?,?,?,?,?,?,?)", (
            int(m["match_id"]), str(m.get("match_date","")),
            str(m.get("home_team","")), str(m.get("away_team","")),
            int(m.get("home_score",0)), int(m.get("away_score",0)),
            "La Liga", "2015/16"
        ))
    con.commit()

    # Load events
    for i, (_, m) in enumerate(matches.iterrows()):
        process_match(con, m["match_id"], m.get("home_team"), m.get("away_team"),
                      m.get("home_score",0), m.get("away_score",0), m.get("match_date",""))
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(matches)} matches processed")

    # Team stats from player stats
    con.execute("DELETE FROM team_match_stats")
    con.execute("""
        INSERT INTO team_match_stats
            (match_id,team_name,goals,shots,shots_on_target,
             passes,pass_accuracy,xg,pressures,result)
        SELECT match_id, team_name,
               sum(goals), sum(shots), sum(shots_on_target),
               sum(passes), round(avg(pass_accuracy),2),
               round(sum(xg),4), sum(pressures),
               'TBD'
        FROM player_match_stats
        GROUP BY match_id, team_name
    """)
    con.commit()

    # Print summary
    print("\n[Pipeline] Summary:")
    for t in ["matches","shots","passes","player_match_stats","team_match_stats"]:
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {n:,} rows")

    con.close()
    return True


if __name__ == "__main__":
    run_pipeline()
