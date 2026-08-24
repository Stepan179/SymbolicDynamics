#!/usr/bin/env python3
"""Карта B: контекстные векторы (target-token pooling) моделью bge-m3.

В трансформер подаётся окно (до 4 reply-предков + 3 соседа по сессии), усредняются
hidden states только по токенам целевого юнита. Батч-токенизация, сортировка по длине,
шардинг; --merge склеивает шарды.

Input:  --units <chat>/units.jsonl, --model <путь к bge-m3>
Output: --out <chat>/vec/{dense_ctx.f16.npy, ids_ctx.json}
Usage:  python3 pipeline/vectorize_ctx2.py --units chats/physics/units.jsonl \
            --out chats/physics/vec --model ~/models/bge-m3 --batch 256
        python3 pipeline/vectorize_ctx2.py --merge chats/physics/vec <nshards>"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

REPLY_ANCESTORS = 4
TEMPORAL_NEIGHBORS = 3
MAXLEN = 512


def build_context(units):
    pos_by_uid = {u["unit_id"]: i for i, u in enumerate(units)}
    msg2unit = {}
    for u in units:
        for mid in u["msg_ids"]:
            msg2unit[mid] = u["unit_id"]

    def line(u):
        return f"{u.get('author') or '?'}: {u['text']}"

    samples = []
    for i, u in enumerate(units):
        if not (u.get("text") or "").strip():
            continue
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
        lines = [line(units[k]) for k in sorted(ctx)]
        who = u.get("author") or "?"
        pre = ("\n".join(lines) + "\n" if lines else "") + f"{who}: "
        samples.append((u["unit_id"], pre, u["text"]))
    return samples


@torch.inference_mode()
def encode_chunk(model, tok, pres, tgts, device):
    """Батч-токенизация + masked mean pool по токенам цели."""
    pre_ids_b = tok(pres, add_special_tokens=False)["input_ids"]
    tgt_ids_b = tok(tgts, add_special_tokens=False)["input_ids"]
    cls, sep = tok.cls_token_id, tok.sep_token_id
    seqs, ranges = [], []
    for pre_ids, tgt_ids in zip(pre_ids_b, tgt_ids_b):
        budget = MAXLEN - 2 - len(tgt_ids)
        if budget < 0:
            tgt_ids = tgt_ids[: MAXLEN - 2]
            pre_ids = []
        elif len(pre_ids) > budget:
            pre_ids = pre_ids[len(pre_ids) - budget:]
        seq = [cls] + pre_ids + tgt_ids + [sep]
        start = 1 + len(pre_ids)
        seqs.append(seq)
        ranges.append((start, start + len(tgt_ids)))

    L = max(len(s) for s in seqs)
    B = len(seqs)
    input_ids = torch.zeros(B, L, dtype=torch.long)
    attn = torch.zeros(B, L, dtype=torch.long)
    pool = torch.zeros(B, L, dtype=torch.float32)
    for k, (seq, (a, b)) in enumerate(zip(seqs, ranges)):
        input_ids[k, : len(seq)] = torch.tensor(seq)
        attn[k, : len(seq)] = 1
        pool[k, a:b] = 1.0
    pool /= pool.sum(1, keepdim=True).clamp(min=1)

    hs = model(input_ids=input_ids.to(device), attention_mask=attn.to(device)).last_hidden_state
    vec = (hs * pool.unsqueeze(-1).to(device)).sum(1)
    vec = torch.nn.functional.normalize(vec, dim=1)
    return vec.float().cpu().numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="data/units.jsonl")
    ap.add_argument("--out", default="data/vec")
    ap.add_argument("--model", default="/mnt/models/bge-m3")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--merge", nargs=2, metavar=("OUT", "N"))
    args = ap.parse_args()

    if args.merge:
        out, n = args.merge[0], int(args.merge[1])
        parts, ids = [], []
        for i in range(n):
            parts.append(np.load(f"{out}/dense_ctx.shard{i}.f16.npy"))
            ids.extend(json.load(open(f"{out}/ids.shard{i}.json")))
        np.save(f"{out}/dense_ctx.f16.npy", np.concatenate(parts))
        json.dump(ids, open(f"{out}/ids_ctx.json", "w"))
        print(f"merged {n} шардов -> {out}/dense_ctx.f16.npy {np.concatenate(parts).shape}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    units = [json.loads(l) for l in open(args.units, encoding="utf-8")]
    samples = build_context(units)
    if args.limit:
        samples = samples[: args.limit]
    samples = samples[args.shard :: args.nshards] if args.nshards > 1 else samples
    ids = [s[0] for s in samples]
    n = len(samples)
    print(f"[shard {args.shard}/{args.nshards}] юнитов: {n}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()

    order = sorted(range(n), key=lambda i: len(samples[i][1]) + len(samples[i][2]))

    def est_tok(i):
        return min(MAXLEN, (len(samples[i][1]) + len(samples[i][2])) // 3 + 2)

    TOKEN_BUDGET = MAXLEN * 40
    MAX_BATCH = args.batch

    os.makedirs(args.out, exist_ok=True)
    suf = f".shard{args.shard}" if args.nshards > 1 else ""
    dat = f"{args.out}/dense_ctx{suf}.f16.dat"
    out = np.memmap(dat, dtype=np.float16, mode="w+", shape=(n, 1024))

    t0 = time.time()
    s = 0
    while s < n:
        e, tok_sum = s, 0
        while e < n and (e - s) < MAX_BATCH:
            t = est_tok(order[e])
            if e > s and tok_sum + t > TOKEN_BUDGET:
                break
            tok_sum += t
            e += 1
        idx = order[s:e]
        pres = [samples[i][1] for i in idx]
        tgts = [samples[i][2] for i in idx]
        vecs = encode_chunk(model, tok, pres, tgts, device)
        for j, i in enumerate(idx):
            out[i] = vecs[j]
        if (s // MAX_BATCH) % 40 == 0 and s:
            r = s / (time.time() - t0)
            out.flush()
            print(f"  {s}/{n}  {r:.0f}/с  (батч {len(idx)})", flush=True)
        s = e
    dt = time.time() - t0
    print(f"[shard {args.shard}] готово {n} за {dt:.1f}с = {n/dt:.0f} юнит/с", flush=True)

    out.flush()
    np.save(f"{args.out}/dense_ctx{suf}.f16.npy", np.array(out))
    del out
    os.remove(dat)
    json.dump(ids, open(f"{args.out}/ids{suf}.json", "w"))

if __name__ == "__main__":
    main()
