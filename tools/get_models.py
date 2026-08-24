#!/usr/bin/env python3
"""Скачивание моделей в каталог вне репозитория.

Output: $VECTORIZE_MODELS или ~/models
Usage:  python3 tools/get_models.py [--dir PATH] [--only bge|llm]
"""
import argparse
import os
import subprocess

BGE_REPO = "https://huggingface.co/BAAI/bge-m3/resolve/main"
BGE_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "sentencepiece.bpe.model", "pytorch_model.bin"]

LLM_URL = ("https://huggingface.co/RefalMachine/RuadaptQwen3-4B-Instruct-GGUF"
           "/resolve/main/Q4_K_M.gguf")
LLM_NAME = "RuadaptQwen3-4B-Instruct-Q4_K_M.gguf"


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"skip {path} ({os.path.getsize(path)/1e6:.0f} MB)")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"get  {url}")
    subprocess.run(["curl", "-L", "-C", "-", "--fail", "-o", path, url], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.environ.get(
        "VECTORIZE_MODELS", os.path.expanduser("~/models")))
    ap.add_argument("--only", choices=["bge", "llm"])
    args = ap.parse_args()

    if args.only != "llm":
        for name in BGE_FILES:
            fetch(f"{BGE_REPO}/{name}", os.path.join(args.dir, "bge-m3", name))
    if args.only != "bge":
        fetch(LLM_URL, os.path.join(args.dir, LLM_NAME))

    print(f"\nmodels dir: {args.dir}")


if __name__ == "__main__":
    main()
