#!/usr/bin/env python3
"""Проверка карты B без утечки: вектор ответа пересчитывается без его предка.

В наивном тесте контекст ответа содержит текст предка, поэтому близость тривиальна.
Здесь предок удаляется из контекста и вектор считается заново.

Input:  --units <chat>/units.jsonl, --vec <chat>/vec, --model <путь к bge-m3>
Output: печать recall@k и MRR без утечки
Usage:  python3 pipeline/eval_ctx_clean.py --units chats/physics/units.jsonl \
            --vec chats/physics/vec --model ~/models/bge-m3"""
import sys
import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, "/mnt/models/tmp")
from vectorize_ctx import encode_batch, REPLY_ANCESTORS, TEMPORAL_NEIGHBORS


def build_context_drop(units, i, drop_uid):
    """Контекст для units[i], но БЕЗ юнита drop_uid (прямого предка)."""
    pos_by_uid = {u["unit_id"]: k for k, u in enumerate(units)}
    msg2unit = {}
    for u in units:
        for mid in u["msg_ids"]:
            msg2unit[mid] = u["unit_id"]

    def line(u):
        return f"{u.get('author') or '?'}: {u['text']}"

    u = units[i]
    ctx = set()
    cur = u
    for _ in range(REPLY_ANCESTORS):
        r = cur.get("reply_to")
        if not r or r not in msg2unit:
            break
        j = pos_by_uid.get(msg2unit[r])
        if j is None or j == i:
            break
        ctx.add(j)
        cur = units[j]
    cnt = 0
    j = i - 1
    while j >= 0 and cnt < TEMPORAL_NEIGHBORS and units[j]["session"] == u["session"]:
        if units[j].get("text"):
            ctx.add(j)
            cnt += 1
        j -= 1
    ctx.discard(i)
    ctx.discard(pos_by_uid.get(drop_uid, -1))
    lines = [line(units[k]) for k in sorted(ctx)]
    who = u.get("author") or "?"
    pre = ("\n".join(lines) + "\n" if lines else "") + f"{who}: "
    return pre, u["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="data/units.jsonl")
    ap.add_argument("--vec", default="data/vec")
    ap.add_argument("--model", default="/mnt/models/bge-m3")
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--pool", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    units = [json.loads(l) for l in open(args.units, encoding="utf-8")]
    pos_by_uid = {u["unit_id"]: k for k, u in enumerate(units)}
    msg2unit = {}
    for u in units:
        for mid in u["msg_ids"]:
            msg2unit[mid] = u["unit_id"]

    dense = np.load(f"{args.vec}/dense_ctx.f16.npy").astype(np.float32)
    ids = json.load(open(f"{args.vec}/ids_ctx.json"))
    row = {uid: k for k, uid in enumerate(ids)}
    dn = dense / (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-9)

    edges = []
    for u in units:
        r = u.get("reply_to")
        if r and r in msg2unit:
            a, b = msg2unit[r], u["unit_id"]
            if a != b and a in row and b in row:
                edges.append((a, b))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(edges)
    edges = edges[: args.pairs]
    print(f"пар для честного теста: {len(edges)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()

    BATCH = 32
    hits10 = 0
    rr = 0.0
    N = len(ids)
    for s in range(0, len(edges), BATCH):
        chunk = edges[s : s + BATCH]
        pairs = [build_context_drop(units, pos_by_uid[b], a) for a, b in chunk]
        q = encode_batch(model, tok, pairs, device).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
        for k, (a, b) in enumerate(chunk):
            ai = row[a]
            cand = rng.integers(0, N, size=args.pool - 1)
            cand = np.append(cand, ai)
            sims = dn[cand] @ q[k]
            order = np.argsort(-sims)
            rank = int(np.where(cand[order] == ai)[0][0]) + 1
            if rank <= 10:
                hits10 += 1
            rr += 1.0 / rank

    n = len(edges)
    print(f"[БЕЗ утечки] Recall@10: {hits10/n:.3f}  (случайно ~{10/args.pool:.3f})")
    print(f"[БЕЗ утечки] MRR:       {rr/n:.3f}")

if __name__ == "__main__":
    main()
