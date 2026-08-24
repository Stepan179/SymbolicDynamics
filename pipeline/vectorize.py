#!/usr/bin/env python3
"""Карта A: dense- и sparse-векторы юнита моделью bge-m3 (юнит сам по себе).

Input:  --units <chat>/units.jsonl, --model <путь к bge-m3>
Output: --out <chat>/vec/{dense.f16.npy, sparse.jsonl, ids.json}
Usage:  python3 pipeline/vectorize.py --units chats/physics/units.jsonl \
            --out chats/physics/vec --model ~/models/bge-m3"""
import os
import sys
import json
import time
import argparse
import numpy as np


def load_units(path, limit):
    units = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            t = (u.get("text") or "").strip()
            if not t:
                continue
            units.append((u["unit_id"], t))
            if limit and len(units) >= limit:
                break
    return units


def decode_sparse(weights, tokenizer, min_weight=0.01):
    """lexical_weights -> {человекочитаемый термин: вес}. Мелочь отсекаем."""
    out = {}
    for tid, w in weights.items():
        w = float(w)
        if w < min_weight:
            continue
        tok = tokenizer.decode([int(tid)]).strip()
        if tok:
            out[tok] = round(max(out.get(tok, 0.0), w), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="data/units.jsonl")
    ap.add_argument("--out", default="data/vec")
    ap.add_argument("--model", default="/mnt/models/bge-m3")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="0 = все юниты")
    ap.add_argument("--no-sparse", action="store_true")
    args = ap.parse_args()

    from FlagEmbedding import BGEM3FlagModel

    os.makedirs(args.out, exist_ok=True)
    units = load_units(args.units, args.limit)
    ids = [u[0] for u in units]
    texts = [u[1] for u in units]
    print(f"юнитов к векторизации: {len(texts)}", flush=True)

    model = BGEM3FlagModel(args.model, use_fp16=True)

    t0 = time.time()
    out = model.encode(
        texts,
        batch_size=args.batch,
        max_length=args.maxlen,
        return_dense=True,
        return_sparse=not args.no_sparse,
        return_colbert_vecs=False,
    )
    dt = time.time() - t0
    print(f"готово за {dt/60:.1f} мин, {len(texts)/dt:.0f} юнит/с", flush=True)

    dense = np.asarray(out["dense_vecs"], dtype=np.float16)
    np.save(os.path.join(args.out, "dense.f16.npy"), dense)
    json.dump(ids, open(os.path.join(args.out, "ids.json"), "w"))
    print(f"dense: {dense.shape} -> {args.out}/dense.f16.npy", flush=True)

    if not args.no_sparse:
        tok = model.tokenizer
        with open(os.path.join(args.out, "sparse.jsonl"), "w", encoding="utf-8") as f:
            for uid, w in zip(ids, out["lexical_weights"]):
                f.write(json.dumps(
                    {"unit_id": uid, "sparse": decode_sparse(w, tok)},
                    ensure_ascii=False) + "\n")
        print(f"sparse -> {args.out}/sparse.jsonl", flush=True)

if __name__ == "__main__":
    main()
