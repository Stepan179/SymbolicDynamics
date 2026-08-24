#!/usr/bin/env python3
"""Проверка эмбеддингов на reply-парах: recall@k и MRR.

Настоящий ответ прячется среди --pool случайных юнитов; измеряется его ранг по косинусу.

Input:  --units <chat>/units.jsonl, --vec <chat>/vec (--dense/--ids задают карту)
Output: печать recall@1/5/10 и MRR
Usage:  python3 pipeline/eval_replies.py --units chats/physics/units.jsonl \
            --vec chats/physics/vec --dense dense_ctx.f16.npy --ids ids_ctx.json"""
import json
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="data/units.jsonl")
    ap.add_argument("--vec", default="data/vec")
    ap.add_argument("--dense", default="dense.f16.npy", help="имя файла векторов (карта A или B)")
    ap.add_argument("--ids", default="ids.json")
    ap.add_argument("--pairs", type=int, default=3000, help="сколько reply-пар проверить")
    ap.add_argument("--pool", type=int, default=1000, help="размер пула кандидатов на пару")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dense = np.load(f"{args.vec}/{args.dense}").astype(np.float32)
    ids = json.load(open(f"{args.vec}/{args.ids}"))
    pos = {uid: i for i, uid in enumerate(ids)}

    msg2unit = {}
    reply_edges = []
    units = [json.loads(l) for l in open(args.units)]
    for u in units:
        for mid in u["msg_ids"]:
            msg2unit[mid] = u["unit_id"]
    for u in units:
        r = u.get("reply_to")
        if r and r in msg2unit:
            child, parent = u["unit_id"], msg2unit[r]
            if child != parent and child in pos and parent in pos:
                reply_edges.append((parent, child))

    rng = np.random.default_rng(args.seed)
    rng.shuffle(reply_edges)
    edges = reply_edges[: args.pairs]
    print(f"reply-пар для теста: {len(edges)} (всего доступно {len(reply_edges)})")

    dn = dense / (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9)

    N = len(ids)
    hits10 = 0
    rr = 0.0
    for a_uid, b_uid in edges:
        ai, bi = pos[a_uid], pos[b_uid]
        cand = rng.integers(0, N, size=args.pool - 1)
        cand = np.append(cand, bi)
        sims = dn[cand] @ dn[ai]
        order = np.argsort(-sims)
        rank = int(np.where(cand[order] == bi)[0][0]) + 1
        if rank <= 10:
            hits10 += 1
        rr += 1.0 / rank

    n = len(edges)
    print(f"Recall@10: {hits10/n:.3f}  (случайно ~{10/args.pool:.3f})")
    print(f"MRR:       {rr/n:.3f}")

if __name__ == "__main__":
    main()
