# Synthetic worst case: 12 concurrent B3 pages (each a distinct pixel-perturbed copy so MM cache
# and prefix cache miss) with 2 crops each (3 images/request), full reviewer prompt, 300 tokens out.
import json, time, concurrent.futures as cf, requests, base64, io
from PIL import Image
from common import *
row = rows()[0]  # B3 2266x3072, 6,778 image tokens
im = Image.open(SET/row['image']).convert('RGB'); W, H = im.size
def variant(i):
    v = im.copy(); px = v.load(); px[i % W, (i * 7) % H] = (i % 256, 0, 0); b = io.BytesIO(); v.save(b, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
def crop_url(i, box):
    b = io.BytesIO(); im.crop(box).save(b, 'PNG'); return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
def one(i):
    body = {"model": MODEL, "max_tokens": 300, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages":[{"role":"user","content":[
                {"type":"text","text":f"Request {i}: review this page and its two crops."},
                {"type":"image_url","image_url":{"url":variant(i)}},
                {"type":"image_url","image_url":{"url":crop_url(i,(0,0,W//2,H//2))}},
                {"type":"image_url","image_url":{"url":crop_url(i,(W//2,H//2,W,H))}},
                {"type":"text","text":prompt_for(row)}]}]}
    t0 = time.time()
    try:
        r = requests.post(URL, json=body, timeout=1800); j = r.json()
        return dict(i=i, status=r.status_code, wall=time.time()-t0, usage=j.get('usage'), err=None if r.status_code==200 else str(j)[:200])
    except Exception as e: return dict(i=i, status=-1, wall=time.time()-t0, usage=None, err=repr(e)[:200])
t0 = time.time()
import sys; N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
with cf.ThreadPoolExecutor(N) as ex: out = list(ex.map(one, range(N)))
ok = [o for o in out if o['status']==200]
print(f"{N} x (B3 + 2 crops): ok {len(ok)}/{N} | wall total {time.time()-t0:.1f}s | per-req wall mean {sum(o['wall'] for o in ok)/max(len(ok),1):.1f}s max {max((o['wall'] for o in ok), default=0):.1f}s")
print("prompt tokens:", sorted(o['usage']['prompt_tokens'] for o in ok if o['usage']))
for o in out:
    if o['status']!=200: print("FAIL", o)
h = requests.get('http://localhost:8000/health', timeout=10).status_code; print("health after:", h)
