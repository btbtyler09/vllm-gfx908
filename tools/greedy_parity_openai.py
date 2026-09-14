# Greedy parity: record on one serve (--record ref.json), compare on another (--compare ref.json).
# 16 text prompts + 4 page images, temperature 0, 96 tokens, top-5 logprobs.
import sys, json, requests
from common import *
TEXT = [
 "Explain why the sky is blue in three sentences.",
 "Write a Python function that returns the n-th Fibonacci number iteratively.",
 "Summarize the causes of the French Revolution.",
 "What is the difference between TCP and UDP?",
 "Translate to French: The library closes at nine on weekdays.",
 "List five prime numbers greater than 100 and explain how you checked one of them.",
 "A train leaves at 3:15 pm and arrives at 6:40 pm. How long is the trip?",
 "Describe the water cycle for a ten-year-old.",
 "Give a haiku about a rusty bicycle.",
 "What does the HTTP status code 429 mean and how should a client respond?",
 "Rewrite this sentence in passive voice: The committee approved the budget.",
 "Explain what a hash table is and one situation where it is a poor choice.",
 "Name three differences between mitosis and meiosis.",
 "Write a SQL query that returns the ten most recent orders per customer.",
 "Why does ice float on water?",
 "Compose a two-line rhyme about coffee.",
]
IMG_ROWS = [1, 5, 10, 13]  # B1, B2, B0, B3
def run(row=None, text=None):
    content = [{"type":"text","text":text}] if row is None else [
        {"type":"image_url","image_url":{"url":data_url(SET/row['image'])}},
        {"type":"text","text":"Transcribe the first three lines of text on this page exactly."}]
    body = {"model": MODEL, "messages":[{"role":"user","content":content}], "max_tokens": 96, "temperature": 0,
            "logprobs": True, "top_logprobs": 5, "chat_template_kwargs": {"enable_thinking": False}}
    j = requests.post(URL, json=body, timeout=600).json()
    ch = j['choices'][0]
    toks = [(c['token'], c['logprob']) for c in ch['logprobs']['content']]
    return {"text": ch['message']['content'], "tokens": toks}
cases = [("text", t) for t in TEXT] + [("img", rows()[i]) for i in IMG_ROWS]
res = []
for kind, x in cases:
    res.append(run(text=x) if kind == "text" else run(row=x))
if sys.argv[1] == "--record":
    json.dump(res, open(sys.argv[2], "w"), ensure_ascii=False, indent=1); print("recorded", len(res), "cases to", sys.argv[2])
else:
    ref = json.load(open(sys.argv[2]))
    ident = 0; first_div = []; lp_diffs = []
    for i, (a, b) in enumerate(zip(ref, res)):
        ta = [t for t, _ in a['tokens']]; tb = [t for t, _ in b['tokens']]
        n = min(len(ta), len(tb)); k = next((j for j in range(n) if ta[j] != tb[j]), None)
        if k is None and len(ta) == len(tb): ident += 1
        else: first_div.append((i, k if k is not None else n))
        for (t1, l1), (t2, l2) in zip(a['tokens'], b['tokens']):
            if t1 != t2: break
            lp_diffs.append(abs(l1 - l2))
    print(f"identical {ident}/{len(res)} | first divergence (case, token) {first_div} | mean |dlogprob| on shared prefix {sum(lp_diffs)/max(len(lp_diffs),1):.5f} max {max(lp_diffs) if lp_diffs else 0:.5f}")
