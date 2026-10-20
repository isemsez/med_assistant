import httpx
import json
import threading
import time

q = "Пациент: " + ("жалобы на одышку при физической нагрузке, "
                   "отёки нижних конечностей к вечеру. " * 55)
body = {"model": "medgemma-answerer", "messages": [
    {"role": "system", "content": "Ты — медицинский ассистент."},
    {"role": "user", "content": q + " Что это может быть?"},
], "temperature": 0, "max_tokens": 2000, "stream": True}
t0 = time.monotonic()
state = {"lines": 0, "err": None}

client = httpx.Client()
req = client.build_request("POST", "http://127.0.0.1:8081/v1/chat/completions",
                           content=json.dumps(body, ensure_ascii=False).encode(),
                           headers={"Content-Type": "application/json"})
r = client.send(req, stream=True)
print("status:", r.status_code)


def closer():
    time.sleep(2.5)
    print(f"[closer] closing at {time.monotonic() - t0:.1f}s")
    r.close()


threading.Thread(target=closer, daemon=True).start()
try:
    for line in r.iter_lines():
        if line.startswith("data:"):
            state["lines"] += 1
except Exception as e:
    state["err"] = f"{type(e).__name__}: {e}"
dt = time.monotonic() - t0
print(f"reader ended at {dt:.1f}s: lines={state['lines']} err={state['err']}")
time.sleep(2)
t1 = time.monotonic()
r2 = httpx.post("http://127.0.0.1:8081/v1/chat/completions",
                content=json.dumps({"model": "medgemma-answerer", "messages": [
                    {"role": "user", "content": "ping"}],
                    "max_tokens": 30, "temperature": 0}).encode(),
                headers={"Content-Type": "application/json"}, timeout=240)
print("followup:", r2.status_code, f"in {time.monotonic() - t1:.1f}s")
