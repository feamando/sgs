"""
Local chat UI for Planck 3.0: stdlib http.server only (no new deps on the box).

    GET  /           the chat page
    POST /api/chat   {"message": "...", "session": "id"} -> turn JSON
    POST /api/reset  {"session": "id"}
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .chat import ChatSession, chat_answer

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Planck 3.0 chat</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#f7f7f5;--fg:#1d1d1f;--mut:#6b6b70;--card:#fff;--acc:#2f5bd3;--line:#e3e3e0}
 @media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececef;--mut:#9a9aa2;--card:#1e1e22;--acc:#7c9cff;--line:#2c2c31}}
 body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
 main{max-width:760px;margin:0 auto;padding:16px;display:flex;flex-direction:column;min-height:100vh;box-sizing:border-box}
 h1{font-size:16px;margin:4px 0 12px}h1 span{color:var(--mut);font-weight:400}
 #log{flex:1;display:flex;flex-direction:column;gap:10px}
 .u{align-self:flex-end;background:var(--acc);color:#fff;padding:8px 12px;border-radius:14px;max-width:80%}
 .b{background:var(--card);border:1px solid var(--line);padding:10px 12px;border-radius:14px;max-width:90%}
 .b blockquote{margin:6px 0;padding-left:10px;border-left:3px solid var(--line);color:var(--mut)}
 .meta{color:var(--mut);font-size:12px;margin-top:6px}
 details{font-size:12px;color:var(--mut);margin-top:4px}
 form{display:flex;gap:8px;position:sticky;bottom:0;background:var(--bg);padding:12px 0}
 input{flex:1;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--fg);font:inherit}
 button{padding:10px 14px;border-radius:10px;border:0;background:var(--acc);color:#fff;font:inherit;cursor:pointer}
 button.g{background:transparent;color:var(--mut);border:1px solid var(--line)}
</style></head><body><main>
<h1>Planck 3.0 <span>· typed decisions, cited answers, local memory</span></h1>
<div id="log"></div>
<form id="f"><input id="q" autocomplete="off" placeholder="Ask, then follow up: 'and H&amp;M?', 'when was it founded?'" autofocus>
<button>Ask</button><button type="button" class="g" id="r">New chat</button></form>
</main><script>
const S = Math.random().toString(36).slice(2), log = document.getElementById('log');
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function md(s){return esc(s).replace(/\\*\\*(.+?)\\*\\*/g,'<b>$1</b>').replace(/^&gt; (.*)$/m,'<blockquote>$1</blockquote>').replace(/\\n/g,'<br>')}
function add(cls, html){const d=document.createElement('div');d.className=cls;d.innerHTML=html;log.appendChild(d);d.scrollIntoView();return d}
document.getElementById('f').onsubmit = async e => {
  e.preventDefault(); const q = document.getElementById('q'); const m = q.value.trim(); if(!m) return;
  q.value=''; add('u', esc(m)); const b = add('b', '<span class="meta">thinking…</span>');
  const r = await fetch('/api/chat',{method:'POST',body:JSON.stringify({message:m,session:S})}).then(r=>r.json());
  const trace = r.trace.map(t=>esc(t)).join(' → ');
  b.innerHTML = md(r.reply) + `<div class="meta">asked: “${esc(r.question)}” · ${r.answer_type} · ${r.rewrite} · ${r.steps} decisions · ${r.web_calls} web calls · ${r.ms} ms</div><details><summary>decision trace</summary>${trace}</details>`;
};
document.getElementById('r').onclick = async () => { await fetch('/api/reset',{method:'POST',body:JSON.stringify({session:S})}); log.innerHTML=''; };
</script></body></html>"""


def make_server(harness_factory, host: str = "127.0.0.1", port: int = 8010):
    sessions: dict[str, ChatSession] = {}
    lock = threading.Lock()  # one harness/model: serialize turns

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/":
                return self._json({"error": "not found"}, 404)
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            sid = str(data.get("session", "default"))
            if self.path == "/api/reset":
                sessions.pop(sid, None)
                return self._json({"ok": True})
            if self.path != "/api/chat":
                return self._json({"error": "not found"}, 404)
            msg = str(data.get("message", "")).strip()[:500]
            with lock:
                sess = sessions.setdefault(sid, ChatSession(harness_factory()))
                t0 = time.perf_counter()
                t = sess.ask(msg)
                ms = int((time.perf_counter() - t0) * 1000)
            r = t.result
            self._json({"reply": chat_answer(t), "question": t.question, "answer_type": t.answer_type,
                        "rewrite": t.rewrite, "steps": r["steps"], "web_calls": r["web_calls"], "ms": ms,
                        "trace": [s["decision"]["action"] + (f" {s['decision']['k']}" if s["decision"]["k"] is not None else "")
                                  for s in r["trajectory"]]})

    return ThreadingHTTPServer((host, port), H)
