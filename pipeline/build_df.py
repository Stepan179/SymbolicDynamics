#!/usr/bin/env python3
"""Сборка таблицы для ноутбуков: юниты + финальные метки + производные колонки.

Input:  <base>/units.jsonl, <base>/topics/labels_final.jsonl
Output: <base>/df.csv
Usage:  python3 pipeline/build_df.py --base chats/physics
"""
import argparse

import pandas as pd

BOT_MEGA = ["Боты и игры", "Боты и служебное"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="chats/physics")
    ap.add_argument("--out")
    args = ap.parse_args()
    base = args.base

    units = pd.read_json(f"{base}/units.jsonl", lines=True)
    labels = pd.read_json(f"{base}/topics/labels_final.jsonl", lines=True)

    df = units.merge(labels, on="unit_id")
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    msg2unit = {m: u for u, ms in zip(units["unit_id"], units["msg_ids"]) for m in ms}
    df["reply_to_unit"] = [msg2unit.get(int(r), -1) if pd.notna(r) else -1
                           for r in df["reply_to"]]
    df["is_bot"] = df["mega"].isin(BOT_MEGA)
    df["year"] = df["ts"].dt.year
    df["tenure_days"] = ((df["ts"] - df.groupby("author_id")["ts"].transform("min"))
                         .dt.total_seconds() / 86400)

    df = df.drop(columns=["msg_ids"])
    out = args.out or f"{base}/df.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} rows, {len(df.columns)} cols -> {out}")


if __name__ == "__main__":
    main()
