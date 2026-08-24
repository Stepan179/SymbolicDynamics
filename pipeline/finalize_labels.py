#!/usr/bin/env python3
"""Сборка финальной разметки: три уровня детализации на каждый юнит.

super (3 класса) <- mega (7 тем + others, от классификатора) + subtopic (50, геометрия).

Input:  <base>/topics/{labels.jsonl, labels_llm.jsonl, topics.json}
Output: <base>/topics/labels_final.jsonl
Usage:  python3 pipeline/finalize_labels.py --base chats/physics"""
import json
import argparse

SUPER = {
    "Баллы и апелляции": "учёба", "Разбор задач": "учёба",
    "Поступление и подготовка": "учёба", "Логистика и быт": "учёба", "Мета": "учёба",
    "Алгоритмы": "учёба", "Курс и отбор": "учёба", "Регионы и доступность": "учёба",
    "Оффтоп/болтовня": "соц", "Боты и игры": "соц",
    "Досуг": "соц", "Модерация": "соц", "Боты и служебное": "соц",
    "others": "others",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data")
    args = ap.parse_args()
    base = args.base

    subname = {t["cluster"]: t["name"] for t in json.load(open(f"{base}/topics/topics.json"))}
    geo = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{base}/topics/labels.jsonl")}
    llm = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{base}/topics/labels_llm.jsonl")}

    from collections import Counter
    lv = {"super": Counter(), "mega": Counter(), "subtopic": Counter()}
    n = 0
    with open(f"{base}/topics/labels_final.jsonl", "w", encoding="utf-8") as f:
        for uid, L in llm.items():
            g = geo.get(uid, {})
            cl = g.get("cluster", -1)
            mega = L["mega"]
            rec = {
                "unit_id": uid,
                "super": SUPER.get(mega, "others"),
                "mega": mega,
                "mega2": L.get("mega2"),
                "subtopic": subname.get(cl) if not g.get("others") else None,
                "subtopic_id": cl if not g.get("others") else -1,
                "conf": L.get("conf"),
                "others": L.get("others"),
                "content_kind": L.get("content_kind"),
                "speech_act": L.get("speech_act"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            lv["super"][rec["super"]] += 1
            lv["mega"][rec["mega"]] += 1
            if rec["subtopic"]:
                lv["subtopic"][rec["subtopic"]] += 1

    print(f"[{base}] {n} сообщений -> {base}/topics/labels_final.jsonl")
    print("  SUPER:", dict(lv["super"].most_common()))
    print("  MEGA :", {k: v for k, v in lv["mega"].most_common()})
    print(f"  SUBTOPIC: {len(lv['subtopic'])} подтем, топ-5: "
          f"{[k for k,_ in lv['subtopic'].most_common(5)]}")

if __name__ == "__main__":
    main()
