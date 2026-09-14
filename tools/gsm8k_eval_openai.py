#!/usr/bin/env python3
"""Self-contained GSM8K accuracy eval against an OpenAI-compatible server.
Measures whether P82's lossy acceptance changes correctness. Thinking mode,
temp=0.6 (Qwen3.6 recommended precise), fixed seed (so strict vs P82 draw the
same uniforms -> the delta is pure P82 effect). Concurrent.

Usage: gsm8k_eval.py <base_url> <served_model> <out.json> [n_questions]
"""
import sys, json, re, concurrent.futures as cf
import requests
from datasets import load_dataset

base, model, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 300

try:
    ds = load_dataset("openai/gsm8k", "main", split="test")
except Exception:
    ds = load_dataset("gsm8k", "main", split="test")
items = [(ds[i]["question"], ds[i]["answer"]) for i in range(min(N, len(ds)))]

def gold_of(ans):
    return int(re.search(r"####\s*(-?[\d,]+)", ans).group(1).replace(",", ""))

def extract(text):
    # use post-thinking content if present
    if "</think>" in text:
        text = text.split("</think>")[-1]
    for pat in [r"answer is\s*\$?\\?\$?(-?[\d,]+)", r"\\boxed\{(-?[\d,]+)\}",
                r"####\s*(-?[\d,]+)", r"=\s*\$?(-?[\d,]+)\b"]:
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            try: return int(m[-1].replace(",", ""))
            except ValueError: pass
    nums = re.findall(r"-?[\d,]+", text)
    for n in reversed(nums):
        try: return int(n.replace(",", ""))
        except ValueError: continue
    return None

def one(i):
    q, a = items[i]
    body = {
        "model": model,
        "messages": [{"role": "user", "content":
            q + "\n\nSolve step by step, then end with 'The answer is <number>'."}],
        "max_tokens": 2048, "temperature": 0.6, "top_p": 0.95, "seed": 1234,
        "extra_body": {"top_k": 20, "min_p": 0.0},
    }
    try:
        r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=300)
        choice = r.json()["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""
        # With a reasoning parser on the serve (e.g. --reasoning-parser qwen3),
        # thinking lands in reasoning_content and a length-capped reply has
        # content None. Score content first; fall back to the thinking text so
        # the result matches parser-less serves, and record where it came from.
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        pred = extract(content) if content else None
        source = "content"
        if pred is None and reasoning:
            pred = extract(reasoning)
            source = "reasoning"
        gold = gold_of(a)
        return {"i": i, "gold": gold, "pred": pred, "correct": pred == gold,
                "len": len(content) + len(reasoning), "source": source,
                "finish": choice.get("finish_reason")}
    except Exception as e:
        return {"i": i, "gold": gold_of(a), "pred": None, "correct": False,
                "err": str(e)[:80]}

results = [None] * len(items)
done = 0
with cf.ThreadPoolExecutor(max_workers=int(__import__("os").environ.get("GSM_CONC", "16"))) as ex:
    futs = {ex.submit(one, i): i for i in range(len(items))}
    for f in cf.as_completed(futs):
        r = f.result(); results[r["i"]] = r; done += 1
        if done % 25 == 0:
            acc = sum(x["correct"] for x in results if x) / done
            print(f"  {done}/{len(items)}  running acc={acc:.3f}", flush=True)

correct = sum(r["correct"] for r in results)
acc = correct / len(results)
json.dump({"model": model, "n": len(results), "correct": correct, "accuracy": acc,
           "results": results}, open(out_path, "w"), indent=1)
from_reason = sum(1 for x in results if x and x.get("source") == "reasoning")
capped = sum(1 for x in results if x and x.get("finish") == "length")
print(f"\nGSM8K accuracy: {correct}/{len(results)} = {acc:.4f}  -> {out_path}")
print(f"  answers taken from reasoning_content: {from_reason}; length-capped replies: {capped}")
