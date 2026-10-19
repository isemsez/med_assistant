"""Tokenize candidate renders and compare against Ollama ground-truth ids.

Dev validation helper for the llama.cpp template port (plan task 1):
reproduces the byte-exact prompts the Ollama modelfile templates rendered,
using llama-tokenize on the stripped GGUF.

Usage: python tools/check_render.py   (from repo root, dev venv)
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKENIZE = ROOT / "engine" / "llamacpp" / "llama-tokenize.exe"
MODEL = ROOT / "models" / "medgemma1.5-4b-q4km-text-only.gguf"


def tok_ids(text):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8", newline="") as f:
        f.write(text)
        path = f.name
    out = subprocess.run(
        [str(TOKENIZE), "-m", str(MODEL), "-f", path, "--ids", "--log-disable"],
        capture_output=True, text=True, check=True).stdout.strip()
    return [int(x) for x in out.strip("[]").split(",")]


def check(name, text, target):
    ids = tok_ids(text)[1:]  # drop BOS
    ok = ids == target
    print(f"{name}: {'MATCH' if ok else 'DIFF'} ({len(ids)} vs {len(target)} tokens)")
    if not ok:
        for i, (a, b) in enumerate(zip(ids, target)):
            if a != b:
                print(f"  first diff at {i}: got {ids[max(0,i-3):i+3]} "
                      f"want {target[max(0,i-3):i+3]}")
                break
        else:
            print(f"  length differs; common prefix "
                  f"{min(len(ids), len(target))} tokens")
    return ok


# Ollama /api/generate ground truth (context arrays, BOS not included)
ANSWERER_GT = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else None

if __name__ == "__main__":
    ans = [
        105, 2430, 90846, 3247, 73244, 236787, 1751, 75538, 768, 4354, 236761,
        1599, 659, 496, 11045, 5526, 16326, 236761, 107, 51426, 50780, 11492,
        237117, 237240, 236881, 106, 236743, 107, 105, 2028, 107]
    lab = [
        105, 2430, 207162, 2192, 43394, 5007, 16046, 1617, 236761, 107, 237941,
        237117, 237240, 106, 236743, 107, 105, 2028, 107, 236820, 36345, 236813,
        108, 954, 36345, 236813, 109]

    ok = True
    # answerer, generate-style: think=true (Ollama default), no system message
    ok &= check("answerer single-turn",
                "<start_of_turn>userSYSTEM INSTRUCTION: think silently if "
                "needed. You are a helpful medical assistant.\n"
                "Что такое ХСН?<end_of_turn>\n<start_of_turn>model\n", ans)
    # labeler: system message + empty think block, v3 tail (3 trailing \n)
    ok &= check("labeler single-turn",
                "<start_of_turn>userТы — система разметки.\n"
                "ХСН<end_of_turn>\n<start_of_turn>model\n"
                "<think>\n\n</think>\n\n\n", lab)
    sys.exit(0 if ok else 1)
