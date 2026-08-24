#!/usr/bin/env python3
"""Роутинг всего корпуса в подтемы через сохранённый PCA(whiten) + KMeans.

Расстояние считается в отбеленном пространстве (сырой косинус к центроидам неинформативен);
уверенность — относительный зазор до второй темы, далёкие юниты уходят в others.

Input:  <base>/vec/{dense_ctx.f16.npy, ids_ctx.json}, <base>/topics/{model.joblib, topics.json}
Output: <base>/topics/labels.jsonl
Usage:  python3 pipeline/label_messages.py --base chats/physics"""
import json
import argparse
import numpy as np
import joblib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data")
    ap.add_argument("--others-pct", type=float, default=95,
                    help="перцентиль d1, выше которого -> others")
    ap.add_argument("--multi-conf", type=float, default=0.10,
                    help="conf ниже -> добавляем 2-ю тему")
    args = ap.parse_args()
    base = args.base

    ids = json.load(open(f"{base}/vec/ids_ctx.json"))
    B = np.load(f"{base}/vec/dense_ctx.f16.npy").astype(np.float32)
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9

    m = joblib.load(f"{base}/topics/model.joblib")
    pca, km, train_mean = m["pca"], m["km"], m["train_mean"]

    P = pca.transform(B - train_mean)
    if m.get("drop_first"):
        P = P[:, 1:]
    D = km.transform(P)
    order = np.argsort(D, axis=1)
    lab = order[:, 0]
    d1 = D[np.arange(len(D)), order[:, 0]]
    d2 = D[np.arange(len(D)), order[:, 1]]
    lab2 = order[:, 1]
    conf = np.clip((d2 - d1) / (d1 + 1e-9), 0, 1)

    thr = np.percentile(d1, args.others_pct)
    others = d1 > thr

    topics = {t["cluster"]: t for t in json.load(open(f"{base}/topics/topics.json"))}
    name = {c: topics[c]["name"] for c in topics}
    mega = {c: topics[c]["mega"] for c in topics}

    cheap = {}
    for l in open(f"{base}/labels_cheap.jsonl"):
        r = json.loads(l)
        cheap[r["unit_id"]] = (r["content_kind"], r["speech_act"])

    out_rows = []
    for i, uid in enumerate(ids):
        c1, c2 = int(lab[i]), int(lab2[i])
        multi = bool(conf[i] < args.multi_conf and not others[i])
        ck, sa = cheap.get(uid, (None, None))
        out_rows.append({
            "unit_id": uid,
            "topic": ("others" if others[i] else name.get(c1, c1)),
            "cluster": (-1 if others[i] else c1),
            "mega": ("others" if others[i] else mega.get(c1)),
            "conf": round(float(conf[i]), 3),
            "topic2": (name.get(c2) if multi else None),
            "cluster2": (c2 if multi else None),
            "multi": multi,
            "others": bool(others[i]),
            "content_kind": ck,
            "speech_act": sa,
        })

    with open(f"{base}/topics/labels.jsonl", "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    n = len(out_rows)
    n_others = sum(r["others"] for r in out_rows)
    n_multi = sum(r["multi"] for r in out_rows)
    megac = Counter(r["mega"] for r in out_rows)
    print(f"[{base}] размечено {n} юнитов -> {base}/topics/labels.jsonl")
    print(f"  others: {n_others} ({n_others/n:.1%}) при пороге d1>p{args.others_pct:.0f}")
    print(f"  мультилейбл: {n_multi} ({n_multi/n:.1%})")
    print(f"  conf: медиана {np.median(conf):.2f}, p90 {np.percentile(conf,90):.2f}")
    print("  распределение по мега-темам:")
    for mg, c in megac.most_common():
        print(f"    {str(mg):26s} {c:6d} ({c/n:4.0%})")

if __name__ == "__main__":
    main()
