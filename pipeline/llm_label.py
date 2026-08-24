#!/usr/bin/env python3
"""LLM-разметка сообщений в мега-темы через OpenAI-совместимый endpoint.

Input:  --seed  JSONL {unit_id, text, context}
        --topics topics.json (таксономия: cluster -> mega)
Output: --out   JSONL {unit_id, megas:[...]}
Usage:  python3 pipeline/llm_label.py --seed seed.jsonl --topics topics.json \
            --out seed_labeled.jsonl [--batch 8] [--sleep 0.6] [--limit N]

Endpoint задаётся через --api (по умолчанию http://127.0.0.1:8080/v1/chat/completions).
Параметр --sleep задаёт паузу между батчами; на слабом или нагруженном хосте
использовать значение больше нуля.
"""
import argparse
import json
import re
import time
import urllib.request

SYSTEM_TEMPLATE = (
    "Ты классифицируешь сообщения из чата по мега-темам. Доступные темы:\n"
    "{taxonomy}\n- others\n\n"
    "ВАЖНО про трактовку тем — понимай их ШИРОКО:\n"
    "• Темы про суть предмета (разбор задач / алгоритмы) покрывают ЛЮБОЕ обсуждение "
    "задач, их условий, решений, методов, формул, самого предмета — ДАЖЕ обрывочную "
    "фразу или спор о физике/коде.\n"
    "• others ставь КРАЙНЕ РЕДКО — только если сообщение вообще ни о чём из списка и не "
    "про жизнь чата/учёбу/олимпиады. Сомневаешься между темой и others — выбирай тему.\n\n"
    "Тебе дают КОНТЕКСТ (предыдущие реплики) и ЦЕЛЕВОЕ сообщение. Классифицируй ТОЛЬКО "
    "целевое, опираясь на контекст. Выбери 1 тему (2 — только если реально о двух). "
    "Отвечай СТРОГО JSON-массивом без пояснений: "
    '[{{"id": <номер>, "megas": ["<тема>", ...]}}]. Названия тем — точно как в списке.'
)


def build_taxonomy(topics):
    mega = {}
    for t in topics:
        mega.setdefault(t["mega"], []).append(t["name"])
    lines = [f"- {m} (напр.: {', '.join(subs[:6])})" for m, subs in mega.items()]
    return list(mega), "\n".join(lines)


def call(api, messages, retries=3):
    body = json.dumps({"model": "local", "messages": messages,
                       "temperature": 0.0, "max_tokens": 700}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                api, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)["choices"][0]["message"]["content"]
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(3)


def parse_json_array(txt):
    if not txt:
        return None
    match = re.search(r"\[.*\]", txt, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except ValueError:
        return None


def format_block(i, record):
    ctx = "\n".join((record.get("context") or "").strip().split("\n")[-5:])[:600]
    head = f"[{i+1}] КОНТЕКСТ:\n{ctx}\n" if ctx else f"[{i+1}] (без контекста)\n"
    return head + f"ЦЕЛЕВОЕ [{i+1}]: {record['text'][:300]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    topics = json.load(open(args.topics, encoding="utf-8"))
    megas, taxonomy = build_taxonomy(topics)
    system = SYSTEM_TEMPLATE.format(taxonomy=taxonomy)

    seed = [json.loads(line) for line in open(args.seed, encoding="utf-8")]
    if args.limit:
        seed = seed[:args.limit]

    out, started = [], time.time()
    for start in range(0, len(seed), args.batch):
        chunk = seed[start:start + args.batch]
        user = "\n\n".join(format_block(i, r) for i, r in enumerate(chunk))
        parsed = parse_json_array(call(args.api, [
            {"role": "system", "content": system},
            {"role": "user", "content": user}]))

        answers = {}
        if parsed:
            for item in parsed:
                try:
                    answers[int(item["id"])] = [
                        m for m in item.get("megas", []) if m in megas or m == "others"]
                except (KeyError, TypeError, ValueError):
                    continue

        for i, record in enumerate(chunk):
            out.append({"unit_id": record["unit_id"],
                        "megas": answers.get(i + 1) or ["others"]})

        with open(args.out, "w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        done = start + len(chunk)
        if (start // args.batch) % 5 == 0:
            rate = done / (time.time() - started + 1e-9)
            print(f"  {done}/{len(seed)}  {rate:.1f} msg/s", flush=True)
        if args.sleep:
            time.sleep(args.sleep)

    labeled = sum(1 for r in out if r["megas"] != ["others"])
    print(f"done: {len(out)} rows, non-others {labeled} ({labeled/len(out):.0%}), "
          f"{(time.time()-started)/60:.1f} min -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
