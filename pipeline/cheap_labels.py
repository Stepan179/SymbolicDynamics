#!/usr/bin/env python3
"""Правиловая разметка типа контента и речевого акта (без LLM).

content_kind: ack | media | short | text; speech_act: question | answer | command |
chatter | statement | media. Дополнительно печатает плотнейшее 90-дневное окно.

Input:  <chat>/units.jsonl
Output: <chat>/labels_cheap.jsonl
Usage:  python3 pipeline/cheap_labels.py chats/physics/units.jsonl chats/physics/labels_cheap.jsonl"""
import re
import sys
import json
import datetime as dt
from collections import Counter

WINDOW_DAYS = 90
SHORT_CHARS = 12

ACKS = {
    "да", "нет", "неа", "ага", "угу", "ок", "окей", "оке", "окда", "лан", "ладно",
    "лол", "кек", "ор", "орну", "ахах", "хах", "хаха", "пхах", "ржу", "ахах",
    "пон", "понял", "поняла", "ясно", "спс", "спасибо", "пасиб", "пожалуйста",
    "плюс", "согл", "согласен", "согласна", "база", "факт", "точно", "верно",
    "хз", "збс", "топ", "кринж", "жиза", "жесть", "омг", "вау", "ух", "оу",
    "ну", "нуу", "эх", "хм", "хмм", "мда", "имба", "гуд", "найс", "ес", "yes",
    "ok", "okay", "lol", "xd", "хд", "лан", "оке", "мб", "нз", "изи", "пф",
    "аминь", "рил", "реально", "воистину", "ес", "ето", "во", "оч", "збc",
}
ACK_PHRASES = {
    "да нет", "ну да", "ну хз", "да ладно", "не знаю", "да уж", "вот вот",
    "и не говори", "как так", "ну такое", "и что", "а что", "ну и", "да да",
    "спасибо большое", "не за что", "всё равно", "да норм", "ну ок",
}

CMD_RE = re.compile(r"^\s*/")
LETTERS_RE = re.compile(r"[^\wа-яёa-z]", re.IGNORECASE)
ELONG_RE = re.compile(r"(.)\1{2,}")
MEDIA_RE = re.compile(r"^\[(фото|видео|голосовое|кружок|стикер|гиф|аудио|файл|опрос|изображение|геолокация|контакт)")


def normalize(t):
    t = t.lower().strip()
    t = ELONG_RE.sub(r"\1", t)
    letters = re.sub(r"[^\wа-яё ]", " ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", letters).strip()


def classify(text, reply_to):
    raw = (text or "").strip()
    if MEDIA_RE.match(raw):
        return "media", "media"
    has_q = "?" in raw
    if CMD_RE.match(raw):
        return "text" if len(raw) > SHORT_CHARS else "short", "command"

    norm = normalize(raw)
    words = norm.split()
    is_ack = (
        not norm
        or norm in ACKS
        or norm in ACK_PHRASES
        or (len(words) <= 2 and all(w in ACKS for w in words) and words)
    )
    if is_ack:
        return "ack", ("question" if has_q else "chatter")

    if len(raw) <= SHORT_CHARS and len(words) <= 2:
        act = "question" if has_q else ("answer" if reply_to else "chatter")
        return "short", act

    act = "question" if has_q else ("answer" if reply_to else "statement")
    return "text", act


def densest_window(dates, days):
    """dates отсортированы. Возвращает (start, end, count) окна шириной days."""
    if not dates:
        return None
    win = dt.timedelta(days=days)
    best = (dates[0], dates[0] + win, 0)
    j = 0
    for i in range(len(dates)):
        while dates[j] < dates[i] - win:
            j += 1
        cnt = i - j + 1
        if cnt > best[2]:
            best = (dates[j], dates[i], cnt)
    return best


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/units.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/labels_cheap.jsonl"

    units = [json.loads(l) for l in open(src, encoding="utf-8")]
    kinds, acts = Counter(), Counter()
    content_dates = []
    with open(out, "w", encoding="utf-8") as f:
        for u in units:
            kind, act = classify(u.get("text", ""), u.get("reply_to"))
            kinds[kind] += 1
            acts[act] += 1
            f.write(json.dumps({
                "unit_id": u["unit_id"],
                "content_kind": kind,
                "speech_act": act,
                "contentful": kind == "text",
            }, ensure_ascii=False) + "\n")
            if kind == "text" and u.get("ts"):
                content_dates.append(dt.datetime.fromisoformat(u["ts"]))

    content_dates.sort()
    win = densest_window(content_dates, WINDOW_DAYS)

    total = len(units)
    print(json.dumps({
        "units_total": total,
        "content_kind": dict(kinds.most_common()),
        "contentful_share": round(kinds["text"] / total, 3),
        "ack_share": round(kinds["ack"] / total, 3),
        "speech_act": dict(acts.most_common()),
    }, ensure_ascii=False, indent=2))
    if win:
        print(f"\nсамое плотное окно {WINDOW_DAYS} дней по СОДЕРЖАТЕЛЬНЫМ:")
        print(f"  {win[0].date()} .. {win[1].date()}  -> {win[2]} content-юнитов")

if __name__ == "__main__":
    main()
