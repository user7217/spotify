"""
Spotify Streaming History Analytics
Usage: python spotify_analytics.py --folder /path/to/history/folder
"""

import json
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# ── load ──────────────────────────────────────────────────────────────────────

def load_history(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(f"{folder}/Streaming_History_Audio_*.json"))
    if not files:
        raise FileNotFoundError(f"No Streaming_History_Audio_*.json in {folder}")

    records = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            records.extend(json.load(fh))

    df = pd.DataFrame(records)
    df = df[df["master_metadata_track_name"].notna()].copy()  # drop podcasts/audiobooks

    df["ts"]             = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
    df["minutes_played"] = df["ms_played"] / 60_000
    df["hours_played"]   = df["ms_played"] / 3_600_000
    df["date"]           = df["ts"].dt.date
    df["hour"]           = df["ts"].dt.hour
    df["dow"]            = df["ts"].dt.day_name()
    df["month"]          = df["ts"].dt.to_period("M")
    df["year"]           = df["ts"].dt.year

    # short play = under 30s (behavioral skip even if skipped=False)
    df["short_play"] = df["ms_played"] < 30_000

    df.rename(columns={
        "master_metadata_track_name":        "track",
        "master_metadata_album_artist_name": "artist",
        "master_metadata_album_album_name":  "album",
    }, inplace=True)

    return df


# ── helpers ───────────────────────────────────────────────────────────────────

def hr(label=""):
    print(f"\n{'─'*60}  {label}")

def show(df, n=20):
    print(df.to_string(index=False, max_rows=n))


# ── analytics ────────────────────────────────────────────────────────────────

def overview(df: pd.DataFrame):
    hr("OVERVIEW")
    total_h   = df["hours_played"].sum()
    total_d   = total_h / 24
    date_min  = df["ts"].min().date()
    date_max  = df["ts"].max().date()
    span_days = (df["ts"].max() - df["ts"].min()).days

    print(f"  Date range          : {date_min} → {date_max}  ({span_days} days)")
    print(f"  Total plays         : {len(df):,}")
    print(f"  Total time          : {total_h:,.1f} h  /  {total_d:,.1f} days")
    print(f"  Unique tracks       : {df['track'].nunique():,}")
    print(f"  Unique artists      : {df['artist'].nunique():,}")
    print(f"  Unique albums       : {df['album'].nunique():,}")
    print(f"  Skipped plays       : {df['skipped'].sum():,}  ({df['skipped'].mean()*100:.1f}%)")
    print(f"  Short plays (<30s)  : {df['short_play'].sum():,}  ({df['short_play'].mean()*100:.1f}%)")
    print(f"  Shuffle plays       : {df['shuffle'].sum():,}  ({df['shuffle'].mean()*100:.1f}%)")
    print(f"  Offline plays       : {df['offline'].sum():,}  ({df['offline'].mean()*100:.1f}%)")


def top_tracks(df: pd.DataFrame, n=25):
    hr("TOP TRACKS — play count")
    t = (df.groupby(["track", "artist"])
           .agg(plays=("ts", "count"), hours=("hours_played", "sum"))
           .sort_values("plays", ascending=False)
           .reset_index()
           .head(n))
    t["hours"] = t["hours"].round(1)
    show(t)

    hr("TOP TRACKS — time listened (hours)")
    t2 = (df.groupby(["track", "artist"])
            .agg(hours=("hours_played", "sum"), plays=("ts", "count"))
            .sort_values("hours", ascending=False)
            .reset_index()
            .head(n))
    t2["hours"] = t2["hours"].round(2)
    show(t2)


def top_artists(df: pd.DataFrame, n=25):
    hr("TOP ARTISTS — play count")
    a = (df.groupby("artist")
           .agg(plays=("ts", "count"), hours=("hours_played", "sum"),
                unique_tracks=("track", "nunique"))
           .sort_values("plays", ascending=False)
           .reset_index()
           .head(n))
    a["hours"] = a["hours"].round(1)
    show(a)


def top_albums(df: pd.DataFrame, n=20):
    hr("TOP ALBUMS — play count")
    a = (df.groupby(["album", "artist"])
           .agg(plays=("ts", "count"), hours=("hours_played", "sum"))
           .sort_values("plays", ascending=False)
           .reset_index()
           .head(n))
    a["hours"] = a["hours"].round(1)
    show(a)


def temporal_patterns(df: pd.DataFrame):
    hr("LISTENING BY HOUR OF DAY")
    h = (df.groupby("hour")
           .agg(plays=("ts", "count"), hours=("hours_played", "sum"))
           .reset_index())
    h["hours"] = h["hours"].round(1)
    h["bar"]   = h["plays"].apply(lambda x: "█" * int(x / h["plays"].max() * 30))
    for _, row in h.iterrows():
        print(f"  {row['hour']:02d}:00  {row['bar']:<32} {row['plays']:>6,} plays  {row['hours']:>6.1f}h")

    hr("LISTENING BY DAY OF WEEK")
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    d = (df.groupby("dow")
           .agg(plays=("ts","count"), hours=("hours_played","sum"))
           .reindex(order)
           .reset_index())
    d["hours"] = d["hours"].round(1)
    show(d)

    hr("LISTENING BY YEAR")
    y = (df.groupby("year")
           .agg(plays=("ts","count"), hours=("hours_played","sum"),
                unique_tracks=("track","nunique"), unique_artists=("artist","nunique"))
           .reset_index())
    y["hours"] = y["hours"].round(1)
    show(y)

    hr("TOP 20 MOST ACTIVE DAYS")
    top_days = (df.groupby("date")
                  .agg(plays=("ts","count"), hours=("hours_played","sum"))
                  .sort_values("hours", ascending=False)
                  .reset_index()
                  .head(20))
    top_days["hours"] = top_days["hours"].round(2)
    show(top_days)


def skip_analysis(df: pd.DataFrame):
    hr("SKIP ANALYSIS — artists with most skips (min 20 plays)")
    a = (df.groupby("artist")
           .agg(plays=("ts","count"), skips=("skipped","sum"),
                short=("short_play","sum"))
           .query("plays >= 20")
           .assign(skip_rate=lambda x: (x["skips"]/x["plays"]*100).round(1))
           .sort_values("skip_rate", ascending=False)
           .reset_index()
           .head(20))
    show(a)

    hr("MOST SKIPPED TRACKS (min 5 plays)")
    t = (df.groupby(["track","artist"])
           .agg(plays=("ts","count"), skips=("skipped","sum"))
           .query("plays >= 5")
           .assign(skip_rate=lambda x: (x["skips"]/x["plays"]*100).round(1))
           .sort_values("skip_rate", ascending=False)
           .reset_index()
           .head(20))
    show(t)

    hr("MOST COMPLETED TRACKS — never skipped (min 10 plays)")
    t2 = (df.groupby(["track","artist"])
            .agg(plays=("ts","count"), skips=("skipped","sum"))
            .query("plays >= 10 and skips == 0")
            .sort_values("plays", ascending=False)
            .reset_index()
            .head(20))
    show(t2)


def session_analysis(df: pd.DataFrame):
    hr("SESSION ANALYSIS")
    s = df.sort_values("ts").copy()
    s["gap_min"] = s["ts"].diff().dt.total_seconds().div(60).fillna(999)
    s["session"] = (s["gap_min"] > 30).cumsum()  # new session if gap > 30 min

    sessions = (s.groupby("session")
                  .agg(start=("ts","min"), end=("ts","max"),
                       plays=("ts","count"), hours=("hours_played","sum"))
                  .assign(duration_h=lambda x:
                      (x["end"]-x["start"]).dt.total_seconds()/3600)
                  .reset_index(drop=True))

    print(f"  Total sessions         : {len(sessions):,}")
    print(f"  Avg plays/session      : {sessions['plays'].mean():.1f}")
    print(f"  Avg session length     : {sessions['duration_h'].mean()*60:.0f} min")
    print(f"  Longest session        : {sessions['duration_h'].max():.2f} h  ({sessions['duration_h'].idxmax()})")

    hr("TOP 10 LONGEST SESSIONS")
    top_s = sessions.sort_values("duration_h", ascending=False).head(10)
    top_s = top_s[["start","end","plays","duration_h"]].copy()
    top_s["duration_h"] = top_s["duration_h"].round(2)
    top_s["start"] = top_s["start"].dt.strftime("%Y-%m-%d %H:%M")
    top_s["end"]   = top_s["end"].dt.strftime("%Y-%m-%d %H:%M")
    show(top_s)


def listening_streaks(df: pd.DataFrame):
    hr("LISTENING STREAKS")
    days = sorted(df["date"].unique())
    days_dt = pd.to_datetime(days)

    max_streak = cur_streak = 1
    max_start  = cur_start  = days_dt[0]
    max_end    = days_dt[0]

    for i in range(1, len(days_dt)):
        if (days_dt[i] - days_dt[i-1]).days == 1:
            cur_streak += 1
            if cur_streak > max_streak:
                max_streak = cur_streak
                max_start  = cur_start
                max_end    = days_dt[i]
        else:
            cur_streak = 1
            cur_start  = days_dt[i]

    total_active = len(days)
    total_span   = (days_dt[-1] - days_dt[0]).days + 1

    print(f"  Active days            : {total_active} / {total_span} ({total_active/total_span*100:.1f}%)")
    print(f"  Longest streak         : {max_streak} days  ({max_start.date()} → {max_end.date()})")


def start_end_reasons(df: pd.DataFrame):
    hr("HOW TRACKS START")
    show(df["reason_start"].value_counts().reset_index()
           .rename(columns={"count":"count"}), n=15)

    hr("HOW TRACKS END")
    show(df["reason_end"].value_counts().reset_index()
           .rename(columns={"count":"count"}), n=15)


def discovery_timeline(df: pd.DataFrame):
    hr("ARTIST DISCOVERY TIMELINE — when you first played each artist (top by plays)")
    first = df.groupby("artist")["ts"].min().reset_index()
    first.columns = ["artist","first_heard"]
    counts = df.groupby("artist").size().reset_index(name="total_plays")
    disc = first.merge(counts).sort_values("total_plays", ascending=False).head(30)
    disc["first_heard"] = disc["first_heard"].dt.strftime("%Y-%m-%d")
    show(disc)


def monthly_trend(df: pd.DataFrame):
    hr("MONTHLY LISTENING TREND (hours)")
    m = (df.groupby("month")
           .agg(hours=("hours_played","sum"), plays=("ts","count"))
           .reset_index())
    m["hours"] = m["hours"].round(1)
    m["bar"]   = m["hours"].apply(lambda x: "█" * int(x / m["hours"].max() * 40))
    for _, row in m.iterrows():
        print(f"  {str(row['month'])}  {row['bar']:<42} {row['hours']:>6.1f}h  {row['plays']:>5,} plays")


def platform_breakdown(df: pd.DataFrame):
    hr("PLATFORM BREAKDOWN")
    show(df["platform"].value_counts().reset_index(), n=15)


def deep_dives(df: pd.DataFrame):
    hr("GENRE PROXY — top artist per year (most hours)")
    yearly = (df.groupby(["year","artist"])
                .agg(hours=("hours_played","sum"))
                .reset_index())
    top_yearly = (yearly.sort_values("hours", ascending=False)
                        .groupby("year").head(3)
                        .sort_values(["year","hours"], ascending=[True,False])
                        .reset_index(drop=True))
    top_yearly["hours"] = top_yearly["hours"].round(1)
    show(top_yearly, n=40)

    hr("LATE NIGHT vs DAYTIME (22:00–04:00 = late night)")
    df["time_slot"] = df["hour"].apply(
        lambda h: "late_night" if h >= 22 or h <= 4 else "daytime")
    ts = (df.groupby("time_slot")
            .agg(plays=("ts","count"), hours=("hours_played","sum"))
            .reset_index())
    ts["hours"] = ts["hours"].round(1)
    show(ts)

    hr("TOP 10 TRACKS DURING LATE NIGHT")
    ln = (df[df["time_slot"]=="late_night"]
            .groupby(["track","artist"])
            .agg(plays=("ts","count"))
            .sort_values("plays", ascending=False)
            .reset_index().head(10))
    show(ln)


# ── main ──────────────────────────────────────────────────────────────────────

def run(folder: str):
    print("Loading history...")
    df = load_history(folder)
    print(f"Loaded {len(df):,} track plays.\n")

    overview(df)
    top_tracks(df)
    top_artists(df)
    top_albums(df)
    temporal_patterns(df)
    monthly_trend(df)
    skip_analysis(df)
    session_analysis(df)
    listening_streaks(df)
    start_end_reasons(df)
    discovery_timeline(df)
    platform_breakdown(df)
    deep_dives(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True,
                        help="Path to folder containing Streaming_History_Audio_*.json")
    args = parser.parse_args()
    run(args.folder)