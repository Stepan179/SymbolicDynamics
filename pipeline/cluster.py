#!/usr/bin/env python3
"""Кластеризация содержательных юнитов окна: PCA(whiten) + KMeans -> подтемы.

Плотностные методы на этих векторах не работают (анизотропия), поэтому whitening + KMeans.
Сохраняет обученный преобразователь для роутинга всего корпуса.

Input:  --vec <chat>/vec, --units <chat>/units.jsonl, --labels <chat>/labels_cheap.jsonl
Output: --out <chat>/topics/{clusters.json, assignments.jsonl, centroids.npy,
        cluster_ids.json, model.joblib}
Usage:  python3 pipeline/cluster.py --vec chats/physics/vec --units chats/physics/units.jsonl \
            --labels chats/physics/labels_cheap.jsonl --out chats/physics/topics \
            --start 2025-01-12 --end 2025-04-11 --method kmeans --k 50"""
import os
import json
import argparse
import datetime as dt
from collections import defaultdict, Counter
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec", default="data/vec")
    ap.add_argument("--units", default="data/units.jsonl")
    ap.add_argument("--labels", default="data/labels_cheap.jsonl")
    ap.add_argument("--out", default="topics")
    ap.add_argument("--start", default="2025-01-12")
    ap.add_argument("--end", default="2025-04-11")
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--method", choices=["kmeans", "hdbscan"], default="kmeans")
    ap.add_argument("--k", type=int, default=50, help="число кластеров для kmeans")
    ap.add_argument("--drop-first-pc", action="store_true",
                    help="выкинуть 1-ю компоненту (анизотропия)")
    ap.add_argument("--min-cluster-size", type=int, default=80)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--reps", type=int, default=12)
    args = ap.parse_args()

    from sklearn.decomposition import PCA
    from sklearn.cluster import HDBSCAN, KMeans

    os.makedirs(args.out, exist_ok=True)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    ids = json.load(open(f"{args.vec}/ids_ctx.json"))
    row_of = {uid: i for i, uid in enumerate(ids)}
    B = np.load(f"{args.vec}/dense_ctx.f16.npy")

    contentful = {json.loads(l)["unit_id"] for l in open(args.labels)
                  if json.loads(l)["content_kind"] == "text"}

    meta = {}
    for l in open(args.units, encoding="utf-8"):
        u = json.loads(l)
        meta[u["unit_id"]] = u

    sel = []
    for uid in ids:
        if uid not in contentful:
            continue
        u = meta.get(uid)
        if not u or not u.get("ts"):
            continue
        d = dt.date.fromisoformat(u["ts"][:10])
        if start <= d <= end:
            sel.append(uid)
    print(f"кластеризуем {len(sel)} содержательных юнитов в окне {start}..{end}", flush=True)

    rows = np.array([row_of[u] for u in sel])
    X = B[rows].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    Xc = X - X.mean(0)

    ncomp = args.pca + (1 if args.drop_first_pc else 0)
    pca = PCA(n_components=ncomp, whiten=True, random_state=0)
    P = pca.fit_transform(Xc)
    if args.drop_first_pc:
        P = P[:, 1:]
    print(f"PCA(whiten) -> {P.shape}", flush=True)

    if args.method == "kmeans":
        km = KMeans(n_clusters=args.k, random_state=0, n_init=10)
        lab = km.fit_predict(P)
        K, noise = args.k, 0
        import joblib
        joblib.dump({"pca": pca, "km": km, "train_mean": X.mean(0),
                     "drop_first": args.drop_first_pc},
                    os.path.join(args.out, "model.joblib"))
        print(f"KMeans k={K}", flush=True)
    else:
        hdb = HDBSCAN(min_cluster_size=args.min_cluster_size,
                      min_samples=args.min_samples, metric="euclidean")
        lab = hdb.fit_predict(P)
        K = len(set(lab)) - (1 if -1 in lab else 0)
        noise = int((lab == -1).sum())
        print(f"кластеров: {K} | шум: {noise} ({noise/len(sel):.1%})", flush=True)

    from sklearn.feature_extraction.text import TfidfVectorizer
    RU_STOP = ("это этот эта эти тот та то те как что чтобы кто где когда чем чём тут там "
               "так тоже также очень просто ещё еще уже вот нет да ну же бы вон вообще "
               "меня тебя себя него неё них его её их мне тебе ему ей нам вам им я ты он она "
               "оно мы вы они был была было были быть есть будет буду если или но а и в во на "
               "с со по за от до из у о об про над под при без для к ко же ли бы то надо можно "
               "нужно потом чтоб раз два всё все весь вся почему потому только даже "
               "сейчас нибудь который которая моё мой моя твой наш ваш этого этом "
               "которые эту тем этой более чуть блин типа короче")
    labels_present = sorted(c for c in set(lab) if c != -1)
    docs = []
    for c in labels_present:
        idx = np.where(lab == c)[0]
        docs.append(" ".join(meta[sel[i]]["text"] for i in idx))
    vec = TfidfVectorizer(
        token_pattern=r"(?u)\b[а-яёА-ЯЁa-zA-Z]{3,}\b",
        stop_words=RU_STOP.split(), max_df=0.5, sublinear_tf=True, lowercase=True)
    tfidf = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())
    ctfidf_top = {}
    for r, c in enumerate(labels_present):
        row = tfidf[r].toarray().ravel()
        ctfidf_top[c] = [terms[j] for j in np.argsort(-row)[:15]]

    clusters = []
    centroids = []
    cluster_ids = []
    for c in sorted(set(lab)):
        if c == -1:
            continue
        idx = np.where(lab == c)[0]
        cuids = [sel[i] for i in idx]
        cvec = X[idx].mean(0)
        cvec /= np.linalg.norm(cvec) + 1e-9
        sims = X[idx] @ cvec
        top = idx[np.argsort(-sims)][: args.reps]
        rep_uids = [sel[i] for i in top]

        keywords = ctfidf_top.get(c, [])

        clusters.append({
            "cluster": int(c),
            "size": len(cuids),
            "keywords": keywords,
            "rep_unit_ids": [int(x) for x in rep_uids],
            "rep_texts": [meta[x]["text"][:200] for x in rep_uids],
        })
        centroids.append(cvec)
        cluster_ids.append(int(c))

    clusters.sort(key=lambda x: -x["size"])
    json.dump(clusters, open(f"{args.out}/clusters.json", "w"), ensure_ascii=False, indent=1)
    np.save(f"{args.out}/centroids.npy", np.array(centroids, dtype=np.float16))
    json.dump(cluster_ids, open(f"{args.out}/cluster_ids.json", "w"))
    with open(f"{args.out}/assignments.jsonl", "w", encoding="utf-8") as f:
        for uid, c in zip(sel, lab):
            f.write(json.dumps({"unit_id": uid, "cluster": int(c)}, ensure_ascii=False) + "\n")

    print(f"\nтоп-кластеры:", flush=True)
    for cl in clusters[:12]:
        print(f"  #{cl['cluster']:3d} n={cl['size']:4d}  {', '.join(cl['keywords'][:6])}", flush=True)
    print(f"\n-> {args.out}/clusters.json, centroids.npy, assignments.jsonl", flush=True)

if __name__ == "__main__":
    main()
