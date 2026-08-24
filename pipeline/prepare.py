#!/usr/bin/env python3
"""Telegram JSON-экспорт -> юниты (единицы анализа) и сообщения.

Заглушки удалённых сообщений, плейсхолдеры медиа, склейка бёрстов одного автора
(<= 60 c, без reply) в юнит, нарезка на сессии (граница дня 05:00, пауза > 30 мин).

Input:  <chat>/source/result.json
Output: <chat>/units.jsonl, <chat>/messages.jsonl
Usage:  python3 pipeline/prepare.py chats/physics/source/result.json chats/physics/"""
import os
import sys
import json
import datetime as dt
from collections import Counter

BURST_SEC = 60
BURST_MAX = 8
SESSION_GAP_MIN = 30
DAY_BOUNDARY_HOUR = 5


MEDIA_LABEL = {
    "sticker": "стикер",
    "animation": "гиф",
    "video_message": "кружок",
    "video_file": "видео",
    "voice_message": "голосовое",
    "audio_file": "аудио",
}


def extract_text(m):
    """text_entities -> плоская строка. Сохраняем меншены/ссылки/хэштеги как текст."""
    ents = m.get("text_entities")
    if ents:
        return "".join(e.get("text", "") for e in ents).strip()
    t = m.get("text")
    if isinstance(t, list):
        return "".join(x if isinstance(x, str) else x.get("text", "") for x in t).strip()
    return (t or "").strip()


def media_kind(m):
    """Возвращает короткий машинный тип медиа или None."""
    if m.get("media_type"):
        return m["media_type"]
    if m.get("photo"):
        return "photo"
    if m.get("poll"):
        return "poll"
    if m.get("location_information"):
        return "location"
    if m.get("contact_information"):
        return "contact"
    if m.get("file"):
        mime = (m.get("mime_type") or "").split("/")[0]
        return "image" if mime == "image" else "file"
    return None


def media_placeholder(m, kind):
    """Человекочитаемая заглушка для сообщения без текста."""
    if kind in MEDIA_LABEL:
        label = MEDIA_LABEL[kind]
        if kind == "sticker" and m.get("sticker_emoji"):
            return f"[{label} {m['sticker_emoji']}]"
        return f"[{label}]"
    if kind == "photo":
        return "[фото]"
    if kind == "image":
        return "[изображение]"
    if kind == "poll":
        q = (m.get("poll") or {}).get("question", "")
        return f"[опрос: {q}]".strip()
    if kind == "location":
        return "[геолокация]"
    if kind == "contact":
        return "[контакт]"
    if kind == "file":
        name = m.get("file_name") or "документ"
        return f"[файл: {name}]"
    return "[вложение]"


def has_spoiler(m):
    return any(e.get("type") == "spoiler" for e in m.get("text_entities", []))


def reaction_count(m):
    return sum(r.get("count", 0) for r in m.get("reactions", []) or [])


def parse_ts(m):
    u = m.get("date_unixtime")
    if u:
        return dt.datetime.fromtimestamp(int(u))
    return dt.datetime.fromisoformat(m["date"]) if m.get("date") else None


def normalize(raw_messages):
    out = []
    for m in raw_messages:
        if m["type"] == "service":
            out.append({
                "id": m["id"], "kind": "service",
                "ts": m.get("date"), "text": extract_text(m) or m.get("action", ""),
            })
            continue
        text = extract_text(m)
        kind = media_kind(m)
        placeholder = media_placeholder(m, kind) if (not text and kind) else None
        out.append({
            "id": m["id"], "kind": "message",
            "ts": m.get("date"),
            "author": m.get("from"), "author_id": m.get("from_id"),
            "reply_to": m.get("reply_to_message_id"),
            "forward_from": m.get("forwarded_from"),
            "text": text,
            "placeholder": placeholder,
            "media": kind,
            "edited": bool(m.get("edited")),
            "spoiler": has_spoiler(m),
            "reactions": reaction_count(m),
        })
    return out


def add_deleted(records):
    ids = [r["id"] for r in records if r["kind"] != "service" and isinstance(r["id"], int) and r["id"] > 0]
    present = {r["id"] for r in records}
    missing = sorted(set(range(min(ids), max(ids) + 1)) - present)
    records.extend({"id": i, "kind": "deleted"} for i in missing)
    records.sort(key=lambda r: r["id"])
    return records, len(missing)


def content_text(r):
    """Что реально пойдёт в вектор: настоящий текст или плейсхолдер медиа."""
    return r["text"] or r.get("placeholder") or ""


def merge_bursts(records):
    """Склейка подряд идущих сообщений одного автора."""
    units = []
    cur = None

    def flush():
        nonlocal cur
        if cur:
            cur["text"] = "\n".join(cur["_texts"])
            del cur["_texts"]
            units.append(cur)
            cur = None

    prev_real = None
    for r in records:
        if r["kind"] == "service":
            flush()
            prev_real = None
            continue
        if r["kind"] == "deleted":
            flush()
            prev_real = r
            continue

        text = content_text(r)
        if not text:
            flush()
            prev_real = r
            continue

        ts = parse_ts_from_iso(r["ts"])
        mergeable = (
            cur is not None
            and r["author_id"] == cur["author_id"]
            and not r["reply_to"]
            and not r["forward_from"]
            and cur["_n"] < BURST_MAX
            and ts and cur["_ts_end"]
            and (ts - cur["_ts_end"]).total_seconds() <= BURST_SEC
            and isinstance(prev_real, dict) and prev_real.get("kind") == "message"
        )
        if mergeable:
            cur["_texts"].append(text)
            cur["msg_ids"].append(r["id"])
            cur["_n"] += 1
            cur["_ts_end"] = ts
            cur["ts_end"] = r["ts"]
            cur["edited"] = cur["edited"] or r["edited"]
            cur["spoiler"] = cur["spoiler"] or r["spoiler"]
            cur["reactions"] += r["reactions"]
            if r["media"]:
                cur["media"].append(r["media"])
        else:
            flush()
            cur = {
                "msg_ids": [r["id"]],
                "author": r["author"], "author_id": r["author_id"],
                "ts": r["ts"], "ts_end": r["ts"],
                "reply_to": r["reply_to"], "forward_from": r["forward_from"],
                "_texts": [text],
                "media": [r["media"]] if r["media"] else [],
                "edited": r["edited"], "spoiler": r["spoiler"],
                "reactions": r["reactions"],
                "_n": 1, "_ts_end": ts,
            }
        prev_real = r
    flush()
    return units


def parse_ts_from_iso(s):
    return dt.datetime.fromisoformat(s) if s else None


def assign_sessions(units):
    """Проставляет day (граница 5 утра) и session (разрыв тишины)."""
    session = 0
    prev_ts = None
    prev_day = None
    for u in units:
        ts = parse_ts_from_iso(u["ts"])
        day = (ts - dt.timedelta(hours=DAY_BOUNDARY_HOUR)).date().isoformat() if ts else None
        new = (
            prev_ts is None
            or day != prev_day
            or (ts - prev_ts).total_seconds() > SESSION_GAP_MIN * 60
        )
        if new:
            session += 1
        u["day"] = day
        u["session"] = session
        prev_ts = ts
        prev_day = day
    return session


def clean_unit(u, idx):
    return {
        "unit_id": idx,
        "msg_ids": u["msg_ids"],
        "n_msgs": len(u["msg_ids"]),
        "author": u["author"],
        "author_id": u["author_id"],
        "ts": u["ts"],
        "ts_end": u["ts_end"],
        "day": u["day"],
        "session": u["session"],
        "reply_to": u["reply_to"],
        "forward_from": u["forward_from"],
        "text": u["text"],
        "media": u["media"] or None,
        "edited": u["edited"],
        "spoiler": u["spoiler"],
        "reactions": u["reactions"],
    }


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "result.json"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "data"
    os.makedirs(outdir, exist_ok=True)

    data = json.load(open(src, encoding="utf-8"))
    records = normalize(data["messages"])
    records, n_deleted = add_deleted(records)

    with open(os.path.join(outdir, "messages.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    units = merge_bursts(records)
    n_sessions = assign_sessions(units)
    units = [clean_unit(u, i) for i, u in enumerate(units)]

    with open(os.path.join(outdir, "units.jsonl"), "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    n_msg = sum(1 for r in records if r["kind"] == "message")
    placeholders = sum(1 for r in records if r["kind"] == "message" and r.get("placeholder"))
    merged = sum(1 for u in units if u["n_msgs"] > 1)
    lens = sorted(len(u["text"]) for u in units)
    media_ph = Counter()
    for r in records:
        if r["kind"] == "message" and r.get("placeholder"):
            media_ph[r["placeholder"].split(":")[0].strip("[]").split()[0]] += 1
    print(json.dumps({
        "messages": n_msg,
        "deleted_placeholders": n_deleted,
        "media_placeholders": placeholders,
        "media_breakdown": dict(media_ph.most_common()),
        "units": len(units),
        "compression": round(len(units) / n_msg, 3),
        "merged_units": merged,
        "unit_median_chars": lens[len(lens) // 2],
        "unit_p90_chars": lens[int(len(lens) * 0.9)],
        "sessions": n_sessions,
        "avg_units_per_session": round(len(units) / n_sessions, 1),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
