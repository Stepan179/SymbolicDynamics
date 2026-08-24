#!/usr/bin/env python3
"""Проверка готовности окружения: пакеты, модели, внешние утилиты, данные.

Печатает статус каждого компонента и итоговую готовность по этапам работы.
Код возврата 0, если готовы все этапы, иначе 1.

Usage:  python3 tools/check_env.py
"""
import importlib.util
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")
MODELS_DIR = os.environ.get("VECTORIZE_MODELS", os.path.join(HOME, "models"))

PACKAGES = {
    "numpy": "notebooks", "pandas": "notebooks", "scipy": "notebooks",
    "sklearn": "notebooks", "seaborn": "notebooks", "matplotlib": "notebooks",
    "joblib": "clustering", "catboost": "notebook 0", "torch": "vectorization",
    "transformers": "vectorization", "FlagEmbedding": "vectorization (sparse)",
    "nbformat": "pdf export", "nbconvert": "pdf export",
}

MODELS = {
    "bge-m3 (embedder)": os.path.join(MODELS_DIR, "bge-m3", "pytorch_model.bin"),
    "RuadaptQwen3-4B (LLM, GGUF)": os.path.join(MODELS_DIR, "RuadaptQwen3-4B-Instruct-Q4_K_M.gguf"),
}

BINARIES = {"llama-server": "LLM labeling", "xelatex": "pdf export",
            "pdftoppm": "pdf export"}

EXTRA_BIN_DIRS = [
    os.path.join(HOME, "Library", "TinyTeX", "bin", "universal-darwin"),
    "/opt/homebrew/bin",
]


def find_binary(name):
    found = shutil.which(name)
    if found:
        return found
    for directory in EXTRA_BIN_DIRS:
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            return candidate
    return None

DATA = [
    "chats/physics/units.jsonl",
    "chats/physics/labels_cheap.jsonl",
    "chats/physics/df.csv",
    "chats/physics/vec/dense_ctx.f16.npy",
    "chats/physics/vec/ids_ctx.json",
    "chats/physics/topics/topics.json",
    "chats/physics/topics/model.joblib",
    "chats/physics/topics/labels_final.jsonl",
    "chats/physics/topics/seed_labeled.jsonl",
    "chats/physics/topics/eval_labeled.jsonl",
    "chats/physics/topics/blockB_meta.csv",
    "chats/physics/topics/blockB_labeled.jsonl",
]


def status(ok):
    return "OK     " if ok else "MISSING"


def size_mb(path):
    return os.path.getsize(path) / 1e6


def main():
    missing = set()

    print("PACKAGES")
    for name, need in PACKAGES.items():
        ok = importlib.util.find_spec(name) is not None
        if not ok:
            missing.add(need)
        print(f"  {status(ok)}  {name:<18} ({need})")

    print("\nMODELS")
    for name, path in MODELS.items():
        ok = os.path.exists(path)
        extra = f"{size_mb(path):.0f} MB" if ok else path
        if not ok:
            missing.add("vectorization" if "bge" in name else "LLM labeling")
        print(f"  {status(ok)}  {name:<28} {extra}")

    print("\nBINARIES")
    for name, need in BINARIES.items():
        path = find_binary(name)
        if path is None:
            missing.add(need)
        print(f"  {status(path is not None)}  {name:<14} ({need})")

    print("\nDATA (chats/physics)")
    for rel in DATA:
        path = os.path.join(ROOT, rel)
        ok = os.path.exists(path)
        if not ok:
            missing.add("notebooks")
        mark = f"{size_mb(path):.1f} MB" if ok else ""
        print(f"  {status(ok)}  {rel:<48} {mark}")

    stages = ["notebooks", "notebook 0", "clustering", "vectorization",
              "vectorization (sparse)", "LLM labeling", "pdf export"]
    print("\nREADINESS")
    for stage in stages:
        ready = stage not in missing
        print(f"  {'READY    ' if ready else 'NOT READY'}  {stage}")

    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
