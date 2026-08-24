#!/usr/bin/env python3
"""Отбор сид-выборки для LLM-разметки: ядра кластеров + пограничные юниты.

К каждому юниту прикладывается контекст (окно как в карте B).

Input:  <base>/topics/labels.jsonl, <base>/units.jsonl
Output: <base>/topics/seed.jsonl
Usage:  python3 pipeline/select_seed.py --base chats/physics"""
import json
import argparse
import random

REPLY_ANCESTORS = 4
TEMPORAL_NEIGHBORS = 3


def build_context_map(units):
    """unit_id -> строка контекста (предки по ответам + соседи по сессии)."""
    pos = {u["unit_id"]: i for i, u in enumerate(units)}
    msg2unit = {}
    for u in units:
        for mid in u["msg_ids"]:
            msg2unit[mid] = u["unit_id"]

    def line(u):
        return f"{u.get('author') or '?'}: {u['text']}"

    ctxmap = {}
    for i, u in enumerate(units):
        idx = set()
        cur = u
        for _ in range(REPLY_ANCESTORS):
            r = cur.get("reply_to")
            if not r or r not in msg2unit:
                break
            j = pos.get(msg2unit[r])
            if j is None or j == i:
                break
            idx.add(j)
            cur = units[j]
        cnt, j = 0, i - 1
        while j >= 0 and cnt < TEMPORAL_NEIGHBORS and units[j]["session"] == u["session"]:
            if units[j].get("text"):
                idx.add(j)
                cnt += 1
            j -= 1
        idx.discard(i)
        ctxmap[u["unit_id"]] = "\n".join(line(units[k]) for k in sorted(idx))
    return ctxmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data")
    ap.add_argument("--per-core", type=int, default=30)
    ap.add_argument("--per-edge", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    base = args.base

    labels = [json.loads(l) for l in open(f"{base}/topics/labels.jsonl")]
    units = [json.loads(l) for l in open(f"{base}/units.jsonl", encoding="utf-8")]
    text = {u["unit_id"]: u.get("text", "") for u in units}
    ctxmap = build_context_map(units)

    by_cluster = {}
    for r in labels:
        if r["content_kind"] != "text" or r["others"]:
            continue
        t = text.get(r["unit_id"], "").strip()
        if len(t) < 8:
            continue
        by_cluster.setdefault(r["cluster"], []).append((r["conf"], r["unit_id"], t))

    seed = []
    for c, rows in by_cluster.items():
        rows.sort(reverse=True)
        core = rows[: args.per-core if False else args.per_core]
        edge = rows[-args.per_edge:] if len(rows) > args.per_core else []
        for _, uid, t in core + edge:
            seed.append({"unit_id": uid, "cluster": int(c),
                         "context": ctxmap.get(uid, ""), "text": t})

    random.shuffle(seed)
    with open(f"{base}/topics/seed.jsonl", "w", encoding="utf-8") as f:
        for r in seed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{base}] сид: {len(seed)} сообщений из {len(by_cluster)} кластеров "
          f"-> {base}/topics/seed.jsonl")

if __name__ == "__main__":
    main()
