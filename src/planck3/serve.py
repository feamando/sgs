"""
Local chat UI for Planck 3.0: stdlib http.server only (no new deps on the box).

    GET  /             the chat page
    GET  /api/digest   "for you" from the local knowledge graph
    POST /api/chat     {"message": "...", "session": "id"} -> turn JSON (answer + depth)
    POST /api/feedback {"kind": "source"|"entity", "target": "...", "value": 1|-1}   (user trust layer)
    POST /api/more     {"session": "id", "domain": "..."}  "more results from here"  (session trust layer)
    POST /api/reset    {"session": "id"}   (also forgets the session trust layer)
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .chat import ChatSession, chat_answer

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Planck 3.0 chat</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#f7f7f5;--fg:#1d1d1f;--mut:#6b6b70;--card:#fff;--acc:#2f5bd3;--line:#e3e3e0;--ok:#1f7a4d;--low:#a15c00;--lowbg:#fff6e6}
 @media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececef;--mut:#9a9aa2;--card:#1e1e22;--acc:#7c9cff;--line:#2c2c31;--ok:#5fd39a;--low:#ffb454;--lowbg:#2a2214}}
 body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
 main{max-width:780px;margin:0 auto;padding:16px;display:flex;flex-direction:column;min-height:100vh;box-sizing:border-box}
 h1{font-size:16px;margin:4px 0 12px}h1 span{color:var(--mut);font-weight:400}
 #log{flex:1;display:flex;flex-direction:column;gap:10px}
 .u{align-self:flex-end;background:var(--acc);color:#fff;padding:8px 12px;border-radius:14px;max-width:80%}
 .b{background:var(--card);border:1px solid var(--line);padding:10px 12px;border-radius:14px;max-width:92%}
 .b.lowc{border-color:var(--low);background:var(--lowbg)}
 .b blockquote{margin:6px 0;padding-left:10px;border-left:3px solid var(--line);color:var(--mut)}
 .tier{display:inline-block;font-size:11px;font-weight:600;letter-spacing:.03em;text-transform:uppercase;padding:1px 7px;border-radius:9px;margin-bottom:4px}
 .tier.confident{color:var(--ok);border:1px solid var(--ok)} .tier.low_confidence{color:var(--low);border:1px solid var(--low)} .tier.none{color:var(--mut);border:1px solid var(--line)}
 .bar{position:relative;height:6px;background:var(--line);border-radius:3px;margin:6px 0 2px;max-width:320px}
 .bar i{position:absolute;left:0;top:0;bottom:0;border-radius:3px;background:var(--acc)} .bar b{position:absolute;top:-3px;width:2px;height:12px;background:var(--fg)}
 .meta{color:var(--mut);font-size:12px;margin-top:6px}
 details{font-size:13px;margin-top:6px} summary{color:var(--acc);cursor:pointer}
 ol{margin:4px 0 4px 18px;padding:0} table{border-collapse:collapse;font-size:12px;margin-top:4px}
 td,th{padding:2px 8px 2px 0;text-align:left;vertical-align:top} th{color:var(--mut);font-weight:500}
 .w{font-variant-numeric:tabular-nums} .pos{color:var(--ok)} .neg{color:var(--low)}
 .tw{overflow-x:auto}
 .ev{margin:6px 0;padding:6px 8px;border-left:3px solid var(--line)} .ev small{color:var(--mut)}
 a{color:var(--acc);text-decoration:none} .src{font-size:12px;color:var(--mut);margin-top:6px}
 .act a{margin-right:8px;font-size:12px}
 form{display:flex;gap:8px;position:sticky;bottom:0;background:var(--bg);padding:12px 0;flex-wrap:wrap}
 input{flex:1;min-width:0;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--fg);font:inherit}
 button{padding:10px 14px;border-radius:10px;border:0;background:var(--acc);color:#fff;font:inherit;cursor:pointer}
 button.g{background:transparent;color:var(--mut);border:1px solid var(--line)}
</style></head><body><main>
<h1>Planck 3.0 <span>· direct answers, how they were collated, how sure it is, local memory</span></h1>
<div id="log"></div>
<form id="f"><input id="q" autocomplete="off" placeholder="Ask, then follow up: 'and H&amp;M?', 'when was it founded?'" autofocus>
<button>Ask</button><button type="button" class="g" id="fy">For you</button><button type="button" class="g" id="r">New chat</button></form>
</main><script>
const S = Math.random().toString(36).slice(2), log = document.getElementById('log');
function esc(s){return String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function md(s){return esc(s).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>').replace(/^&gt; (.*)$/m,'<blockquote>$1</blockquote>').replace(/\n/g,'<br>')}
function add(cls, html){const d=document.createElement('div');d.className=cls;d.innerHTML=html;log.appendChild(d);d.scrollIntoView();return d}
function pct(x){return x==null ? 'n/a' : Math.round(x*100)+'%'}
function num(x){if(typeof x!=='number') return esc(x); const c=x>0?'pos':x<0?'neg':''; return `<span class="w ${c}">${x>0?'+':''}${x.toFixed(2)}</span>`}
const LABEL = {confident:'Confident', low_confidence:'Low confidence in my results', none:'No answer found',
  high:'High confidence', good:'Good confidence, might have bias', low:'Low confidence in my results', sceptical:'Treat with scepticism'};
const TIERCLS = {high:'confident', good:'confident', low:'low_confidence', sceptical:'low_confidence', none:'none', confident:'confident', low_confidence:'low_confidence'};
function trustRow(s){ return trustRow5(s); }   // one trust record shape since round 5
function trustRow5(s){
  const t = s.trust, d = esc(s.domain), you = t.personal==null ? '–' : t.personal.toFixed(0);
  const flag = t.conflict ? ' <span class="tier low_confidence" title="your score is 3+ points from the system score">you disagree</span>' : '';
  return `<tr><td>${esc(s.role)}</td><td>${s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${d}</a>` : d}${flag}</td>
   <td class="w">${t.system.toFixed(0)}</td><td class="w">${you}</td><td class="w"><b>${t.effective.toFixed(1)}</b></td>
   <td title="${esc(t.why)}">${esc(t.category)}</td>
   <td class="act"><a href="#" data-fb="1" data-d="${d}">trust more</a><a href="#" data-fb="-1" data-d="${d}">trust less</a><a href="#" data-more="${d}">more from here</a></td></tr>`;
}
function explain5(ex){
  let h = `<ol>${ex.steps.map(s=>`<li>${esc(s)}</li>`).join('')}</ol>`;
  if(ex.items?.length) h += `<div class="src"><b>Confidence ${ex.confidence10.toFixed(1)}/10</b> (${esc(ex.band_label)}), point by point:</div><table>` + ex.items.map(([k,v])=>`<tr><td>${num(v)}</td><td>${esc(k)}</td></tr>`).join('') + '</table>';
  if(ex.candidates?.length) h += '<div class="src"><b>Candidates</b> (reader p · best source trust): ' + ex.candidates.map(c=>`${esc(c.value)} (${c.r.toFixed(2)} · ${(c.top_trust||0).toFixed(0)}/10)`).join(' · ') + '</div>';
  h += `<div class="src">reader: ${esc(ex.reader)} · writer: ${esc(ex.writer)}${ex.writer_fallback ? ' (fell back to the template: '+esc(ex.writer_problems.slice(0,2).join('; '))+')' : ''}</div>`;
  if(ex.sources?.length) h += '<div class="src"><b>Sources</b>: trust 0-10, system · you · effective (you count for at most 30%)</div><div class="tw"><table><tr><th>role</th><th>source</th><th>system</th><th>you</th><th>effective</th><th>category</th><th></th></tr>' + ex.sources.map(trustRow5).join('') + '</table></div>';
  return h;
}
function explainHtml(ex){
  if(!ex) return '';
  if(ex.pipeline === 'answer') return explain5(ex);
  let h = `<ol>${ex.steps.map(s=>`<li>${esc(s)}</li>`).join('')}</ol>`;
  if(ex.candidates?.length) h += '<div class="src"><b>Candidates it chose between:</b> ' + ex.candidates.map(c=>esc(c.value)+(c.p!=null?` (${pct(c.p)})`:` (score ${(c.score||0).toFixed(2)})`)).join(' · ') + '</div>';
  if(ex.why_value?.length) h += '<div class="src"><b>Why this value</b> (score terms):</div><table>' + ex.why_value.map(([k,v])=>`<tr><td>${esc(k)}</td><td>${num(v)}</td></tr>`).join('') + '</table>';
  const wc = ex.why_confidence || {};
  if(wc.items?.length) h += `<div class="src"><b>Why this confidence</b> (${esc(wc.kind)}):</div><table>` + wc.items.map(([k,v])=>`<tr><td>${esc(k)}</td><td>${num(v)}</td></tr>`).join('') + '</table>';
  if(ex.sources?.length) h += '<div class="src"><b>Sources</b>: trust 0-10, system · you · effective</div><div class="tw"><table><tr><th>role</th><th>source</th><th>system</th><th>you</th><th>effective</th><th>category</th><th></th></tr>' + ex.sources.map(trustRow).join('') + '</table></div>';
  return h;
}
function depthHtml(d){
  const ev = d.passages.map(p=>`<div class="ev">${esc(p.text)}<br><small>${esc(p.domain)} · trust ${p.trust.toFixed(2)} · <a href="${esc(p.url)}" target="_blank" rel="noopener">open</a> · <a href="#" data-more="${esc(p.domain)}">more from here</a></small></div>`).join('');
  const rel = d.related.length ? '<div class="src"><b>Also in memory:</b> ' + d.related.map(f=>esc(f.question+' '+f.value)).join(' · ') + '</div>' : '';
  const more = d.sources.filter(s=>!s.read).slice(0,5).map(s=>`<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title||s.domain)}</a>${s.session?' (this chat asked for more)':''}`).join(' · ');
  return `${ev}${rel}${more ? '<div class="src"><b>Further reading:</b> '+more+'</div>' : ''}`;
}
document.getElementById('f').onsubmit = async e => {
  e.preventDefault(); const q = document.getElementById('q'); const m = q.value.trim(); if(!m) return;
  q.value=''; add('u', esc(m)); const b = add('b', '<span class="meta">thinking…</span>');
  const r = await fetch('/api/chat',{method:'POST',body:JSON.stringify({message:m,session:S})}).then(r=>r.json());
  const ex = r.explain || {tier:'none'}, d = r.depth, cls = TIERCLS[ex.tier] || 'none';
  if(cls==='low_confidence') b.classList.add('lowc');
  let meter = '';
  if(ex.pipeline==='answer' && ex.confidence10!=null){
    meter = `<div class="bar" title="confidence ${ex.confidence10}/10"><i style="width:${ex.confidence10*10}%"></i><b style="left:70%"></b></div><div class="meta">confidence ${ex.confidence10.toFixed(1)}/10 · ${esc(ex.band_label)} · ${esc(ex.qtype.replace('_',' '))} question</div>`;
    if(ex.divergence) meter += `<div class="b lowc" style="margin-top:6px">Your source preferences changed this answer. Without them: <b>${esc(ex.divergence.system_value)}</b> (${ex.divergence.system_confidence}/10, ${esc(ex.divergence.system_band)}).</div>`;
  } else if(ex.confidence!=null) {
    meter = `<div class="bar" title="confidence ${pct(ex.confidence)} · bar ${pct(ex.bar)}"><i style="width:${Math.round(ex.confidence*100)}%"></i><b style="left:${Math.round(ex.bar*100)}%"></b></div><div class="meta">confidence ${pct(ex.confidence)} (${esc(ex.confidence_kind)}) · bar ${pct(ex.bar)}</div>`;
  }
  const how = `<details ${cls!=='confident'?'open':''}><summary>How I got this</summary>${explainHtml(r.explain)}</details>`;
  const depth = `<details ${ex.tier==='none'?'open':''}><summary>In depth: ${d.passages.length} passages from ${new Set(d.passages.map(p=>p.url)).size} sources</summary>${depthHtml(d)}</details>`;
  b.innerHTML = `<span class="tier ${cls}">${LABEL[ex.tier] || ''}</span><br>` + md(r.reply) + meter + how + depth +
    `<div class="meta">asked: “${esc(r.question)}” · ${r.answer_type} · ${r.rewrite} · ${r.steps} decisions · ${r.web_calls} web calls · ${r.ms} ms</div>`;
};
log.addEventListener('click', async e => {
  const a = e.target.closest('a[data-fb],a[data-more]'); if(!a) return; e.preventDefault();
  if(a.dataset.fb){
    await fetch('/api/feedback',{method:'POST',body:JSON.stringify({kind:'source',target:a.dataset.d,value:+a.dataset.fb})});
    a.textContent = +a.dataset.fb>0 ? 'trusted more ✓' : 'trusted less ✓'; return;
  }
  a.textContent = 'reading…';
  const r = await fetch('/api/more',{method:'POST',body:JSON.stringify({session:S,domain:a.dataset.more})}).then(r=>r.json());
  a.textContent = 'more from here ✓';
  if(r.error){ add('b', `<span class="meta">${esc(r.error)}</span>`); return; }
  const t = r.trust, tr = t.effective!=null ? `trust ${t.effective.toFixed(1)}/10` : `trust ${t.combined.toFixed(2)}`;
  add('b', `<b>More from ${esc(r.domain)}</b> <span class="meta">${tr} (unchanged: this button fetches more, it does not change trust); later questions in this chat also search it</span>` +
    (r.passages.length ? r.passages.map(p=>`<div class="ev">${esc(p.text)}<br><small><a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title||p.domain)}</a></small></div>`).join('') : '<div class="meta">nothing more on this question from there</div>'));
});
document.getElementById('fy').onclick = async () => {
  const d = await fetch('/api/digest').then(r=>r.json()); const g = d.graph;
  let h = `<b>For you</b> <span class="meta">${g.facts} facts · ${g.passages} passages · ${g.entities} entities · ${g.queries} questions · trusted: ${esc(d.trusted.slice(0,8).join(', ')||'none yet')}</span>`;
  if(!d.interests.length) h += '<br>Nothing yet: ask a few questions first.';
  for(const s of d.interests){
    h += `<div class="ev"><b>${esc(s.entity)}</b>` + s.recent.map(p=>`<br>${esc(p.text.slice(0,240))} <small>(${esc(p.domain)})</small>`).join('');
    if(s.adjacent.length) h += '<br><small>Adjacent: ' + s.adjacent.map(a=>esc(a.entity)).join(' · ') + '</small>';
    h += '</div>';
  }
  if(d.stale.length) h += '<div class="src"><b>Due for a refresh:</b> ' + d.stale.map(f=>esc(f.question)).join(' · ') + '</div>';
  add('b', h);
};
document.getElementById('r').onclick = async () => { await fetch('/api/reset',{method:'POST',body:JSON.stringify({session:S})}); log.innerHTML=''; };
</script></body></html>"""


def make_server(harness_factory, host: str = "127.0.0.1", port: int = 8010):
    sessions: dict[str, ChatSession] = {}
    store = harness_factory().store  # one shared personal knowledge graph
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
            if self.path == "/api/digest":
                from .digest import build_digest
                with lock:
                    return self._json(build_digest(store))
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
            if self.path == "/api/feedback":
                with lock:
                    store.feedback(str(data.get("kind", "source")), str(data.get("target", "")),
                                   int(data.get("value", 1)))
                return self._json({"ok": True})
            if self.path == "/api/reset":
                sessions.pop(sid, None)
                return self._json({"ok": True})
            if self.path == "/api/more":
                with lock:
                    sess = sessions.get(sid)
                    if not sess or not sess.turns:
                        return self._json({"error": "ask something first"})
                    return self._json(sess.h.more_from(str(data.get("domain", "")), sess.turns[-1].question))
            if self.path != "/api/chat":
                return self._json({"error": "not found"}, 404)
            msg = str(data.get("message", "")).strip()[:500]
            with lock:
                sess = sessions.setdefault(sid, ChatSession(harness_factory()))
                t0 = time.perf_counter()
                t = sess.ask(msg)
                ms = int((time.perf_counter() - t0) * 1000)
            r = t.result
            self._json({"reply": chat_answer(t), "depth": r.get("depth", {"passages": [], "sources": [], "related": []}),
                        "question": t.question, "answer_type": t.answer_type,
                        "rewrite": t.rewrite, "steps": r["steps"], "web_calls": r["web_calls"], "ms": ms,
                        "explain": r.get("explain"), "tier": r.get("tier"),
                        "trace": [s["decision"]["action"] + (f" {s['decision']['k']}" if s["decision"]["k"] is not None else "")
                                  for s in r["trajectory"]]})

    return ThreadingHTTPServer((host, port), H)
