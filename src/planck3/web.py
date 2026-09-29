"""
Deterministic web tools: search backends, polite fetch, main-text extraction.

Every search result and fetched page is cached on disk (sha1 of the key), so a
benchmark run can be replayed bit-identically after the live web changes.

Net modes:
    live     use cache if present, otherwise hit the network and cache it (default)
    replay   cache only; a miss is an empty result (the eval of record)
    refresh  always hit the network, overwrite the cache
"""

import hashlib
import html as html_lib
import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib import robotparser

from .util import append_jsonl, now_iso

USER_AGENT = "Planck3-research/0.1 (+https://github.com/feamando/sgs; personal research)"
FETCH_TIMEOUT = 15
MAX_HTML_BYTES = 3_000_000
MIN_SECONDS_PER_HOST = 1.0


class WebCache:
    def __init__(self, root: str | Path, mode: str = "live"):
        assert mode in ("live", "replay", "refresh"), mode
        self.root = Path(root)
        self.mode = mode
        (self.root / "pages").mkdir(parents=True, exist_ok=True)
        (self.root / "search").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def _path(self, kind: str, key: str) -> Path:
        return self.root / kind / f"{self._key(key)}.json"

    def get(self, kind: str, key: str):
        if self.mode == "refresh":
            return None
        p = self._path(kind, key)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def put(self, kind: str, key: str, obj: dict):
        p = self._path(kind, key)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    @property
    def online(self) -> bool:
        return self.mode != "replay"


class Web:
    """Search + fetch with cache, robots.txt and per-host rate limiting."""

    def __init__(self, cache: WebCache, search_backend: str = "searxng",
                 searxng_url: str = "http://localhost:8888", n_results: int = 8,
                 log_path: str | Path | None = None):
        import requests
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.cache = cache
        self.backend = search_backend
        self.searxng_url = searxng_url.rstrip("/")
        self.n_results = n_results
        self.log_path = log_path
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self._last_hit: dict[str, float] = {}
        self.calls = {"search": 0, "fetch": 0, "search_net": 0, "fetch_net": 0}

    # ── search ───────────────────────────────────────────────────────────
    def search(self, query: str) -> list[dict]:
        """Return [{url, title, snippet, domain}] (at most n_results)."""
        self.calls["search"] += 1
        key = f"{self.backend}|{query}"
        hit = self.cache.get("search", key)
        if hit is not None:
            return hit["results"][: self.n_results]
        if not self.cache.online:
            return []
        self.calls["search_net"] += 1
        try:
            if self.backend == "searxng":
                results = self._search_searxng(query)
            elif self.backend == "wikipedia":
                results = self._search_wikipedia(query)
            else:
                raise ValueError(f"unknown search backend {self.backend}")
        except Exception as e:  # network errors must not kill a benchmark run
            self._log({"event": "search_error", "query": query, "error": repr(e)})
            return []
        self.cache.put("search", key, {"query": query, "backend": self.backend,
                                       "at": now_iso(), "results": results})
        return results[: self.n_results]

    def _search_searxng(self, query: str) -> list[dict]:
        r = self.session.get(f"{self.searxng_url}/search",
                             params={"q": query, "format": "json", "language": "en"},
                             timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        out, seen = [], set()
        for item in r.json().get("results", []):
            url = item.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            out.append({"url": url, "title": item.get("title", ""),
                        "snippet": item.get("content", "") or "",
                        "domain": domain_of(url)})
        return out[:20]

    def _search_wikipedia(self, query: str) -> list[dict]:
        r = self.session.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": 10}, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
        out = []
        for item in r.json().get("query", {}).get("search", []):
            title = item["title"]
            url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
            snippet = html_to_text(item.get("snippet", ""))
            out.append({"url": url, "title": title, "snippet": snippet,
                        "domain": "en.wikipedia.org"})
        return out

    def searxng_alive(self) -> bool:
        try:
            r = self.session.get(f"{self.searxng_url}/search",
                                 params={"q": "test", "format": "json"}, timeout=5)
            return r.status_code == 200 and "results" in r.json()
        except Exception:
            return False

    # ── fetch ────────────────────────────────────────────────────────────
    def fetch(self, url: str) -> dict:
        """Return {url, status, html, fetched_at}; status 0 = not fetched."""
        self.calls["fetch"] += 1
        hit = self.cache.get("pages", url)
        if hit is not None:
            return hit
        if not self.cache.online:
            return {"url": url, "status": 0, "html": "", "fetched_at": None, "error": "replay-miss"}
        if not self._robots_ok(url):
            page = {"url": url, "status": 0, "html": "", "fetched_at": now_iso(), "error": "robots"}
            self.cache.put("pages", url, page)
            return page
        self._rate_limit(url)
        self.calls["fetch_net"] += 1
        try:
            r = self.session.get(url, timeout=FETCH_TIMEOUT, stream=True)
            ctype = r.headers.get("content-type", "")
            body = r.raw.read(MAX_HTML_BYTES, decode_content=True) if "html" in ctype else b""
            enc = r.encoding or "utf-8"
            page = {"url": url, "final_url": r.url, "status": r.status_code,
                    "html": body.decode(enc, errors="replace"), "fetched_at": now_iso(),
                    "content_type": ctype}
        except Exception as e:
            page = {"url": url, "status": 0, "html": "", "fetched_at": now_iso(), "error": repr(e)}
        self.cache.put("pages", url, page)
        return page

    def _robots_ok(self, url: str) -> bool:
        host = urlparse(url).scheme + "://" + urlparse(url).netloc
        if host not in self._robots:
            rp = robotparser.RobotFileParser()
            try:
                r = self.session.get(host + "/robots.txt", timeout=5)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except Exception:
                rp.parse([])
            self._robots[host] = rp
        rp = self._robots[host]
        return rp.can_fetch(USER_AGENT, url) if rp else True

    def _rate_limit(self, url: str):
        host = urlparse(url).netloc
        wait = MIN_SECONDS_PER_HOST - (time.time() - self._last_hit.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        self._last_hit[host] = time.time()

    def _log(self, obj):
        if self.log_path:
            append_jsonl(self.log_path, obj)


# ── text extraction ──────────────────────────────────────────────────────
def domain_of(url: str) -> str:
    d = urlparse(url).netloc.lower()
    return d[4:] if d.startswith("www.") else d


def html_to_text(html: str) -> str:
    """Crude tag stripper (fallback + snippets)."""
    html = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h[1-6]|td|th)>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def extract_main_text(html: str, url: str = "") -> str:
    """Main content via trafilatura (tables kept: infoboxes carry facts); crude fallback."""
    if not html:
        return ""
    try:
        import trafilatura
        text = trafilatura.extract(html, url=url or None, include_tables=True,
                                   include_comments=False, favor_recall=True)
        if text and len(text) > 200:
            return text
    except ImportError:
        pass
    except Exception:
        pass
    return html_to_text(html)


def extract_jsonld(html: str) -> list[str]:
    """Flatten schema.org JSON-LD into 'key: value' lines (prices, hours, addresses)."""
    lines = []
    for block in re.findall(r'(?is)<script[^>]+application/ld\+json[^>]*>(.*?)</script>', html or ""):
        try:
            data = json.loads(block.strip())
        except Exception:
            continue
        _flatten(data, "", lines)
    return lines[:200]


def _flatten(obj, prefix, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("@") and k != "@type":
                continue
            _flatten(v, f"{prefix}{k}." if prefix or k != "@graph" else "", out)
    elif isinstance(obj, list):
        for v in obj[:20]:
            _flatten(v, prefix, out)
    elif isinstance(obj, (str, int, float)) and str(obj).strip():
        val = str(obj).strip()
        if len(val) < 200 and not val.startswith("http"):
            out.append(f"{prefix.rstrip('.')}: {val}")
