#!/usr/bin/env python3
"""Классификатор мега-тем: обучение на LLM-размеченном сиде, разметка всего корпуса.

OneVsRest logistic regression по векторам карты B; качество оценивается 5-fold CV на сиде
(оценка оптимистична, честная — на отложенном наборе).

Input:  <base>/vec/{dense_ctx.f16.npy, ids_ctx.json}, <base>/topics/seed_labeled.jsonl
Output: <base>/topics/labels_llm.jsonl
Usage:  python3 pipeline/train_student.py --base chats/physics"""
import json
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data")
    ap.add_argument("--thr", type=float, default=0.35, help="порог вероятности класса")
    ap.add_argument("--multi-thr", type=float, default=0.35)
    args = ap.parse_args()
    base = args.base

    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import f1_score, classification_report

    ids = json.load(open(f"{base}/vec/ids_ctx.json"))
    row = {u: i for i, u in enumerate(ids)}
    B = np.load(f"{base}/vec/dense_ctx.f16.npy").astype(np.float32)
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9

    seed = [json.loads(l) for l in open(f"{base}/topics/seed_labeled.jsonl")]
    seed = [s for s in seed if s["unit_id"] in row]
    Xs = B[[row[s["unit_id"]] for s in seed]]
    Ys_raw = [s["megas"] for s in seed]

    mlb = MultiLabelBinarizer()
    Ys = mlb.fit_transform(Ys_raw)
    classes = list(mlb.classes_)
    print(f"[{base}] сид {len(seed)} | классы: {classes}")

    clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced"))

    yp = cross_val_predict(clf, Xs, Ys, cv=5, method="predict")
    print(f"  cross-val micro-F1: {f1_score(Ys, yp, average='micro'):.3f} | "
          f"macro-F1: {f1_score(Ys, yp, average='macro'):.3f}")
    print(classification_report(Ys, yp, target_names=classes, zero_division=0, digits=2))

    clf.fit(Xs, Ys)
    P = clf.predict_proba(B)
    order = np.argsort(-P, axis=1)
    c1 = order[:, 0]
    p1 = P[np.arange(len(P)), c1]
    c2 = order[:, 1]
    p2 = P[np.arange(len(P)), c2]
    others_idx = classes.index("others") if "others" in classes else -1

    cheap = {}
    for l in open(f"{base}/labels_cheap.jsonl"):
        r = json.loads(l)
        cheap[r["unit_id"]] = (r["content_kind"], r["speech_act"])

    out = []
    for i, uid in enumerate(ids):
        is_other = (p1[i] < args.thr) or (c1[i] == others_idx)
        mega = "others" if is_other else classes[c1[i]]
        mega2 = classes[c2[i]] if (not is_other and p2[i] >= args.multi_thr
                                   and c2[i] != others_idx) else None
        ck, sa = cheap.get(uid, (None, None))
        out.append({"unit_id": uid, "mega": mega, "mega2": mega2,
                    "conf": round(float(p1[i]), 3), "others": bool(is_other),
                    "content_kind": ck, "speech_act": sa})

    with open(f"{base}/topics/labels_llm.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    dist = Counter(r["mega"] for r in out)
    n = len(out)
    nmulti = sum(1 for r in out if r["mega2"])
    print(f"\n  размечено {n} -> {base}/topics/labels_llm.jsonl")
    print(f"  мультилейбл: {nmulti} ({nmulti/n:.0%})")
    for m, c in dist.most_common():
        print(f"    {m:26s} {c:6d} ({c/n:4.0%})")

if __name__ == "__main__":
    main()
