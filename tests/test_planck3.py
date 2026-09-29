"""Offline tests for src/planck3 (no network, no GPU). Run: python -m pytest tests/test_planck3.py -q"""

import bz2
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.planck3.actions import Decision, InvalidDecision, validate  # noqa: E402
from src.planck3.candidates import question_unit, span_candidates  # noqa: E402
from src.planck3.harness import Harness, render_card  # noqa: E402
from src.planck3.metrics import ece, is_correct, summarize  # noqa: E402
from src.planck3.policies import HeuristicPolicy, LLMPolicy, parse_decision, render_observation  # noqa: E402
from src.planck3.store import DAY, Store, ttl_for  # noqa: E402


# ── fake web ─────────────────────────────────────────────────────────────
PAGES = {
    "https://example.org/ikea": "<html><body><h1>IKEA</h1><p>IKEA is a Swedish furniture retailer. "
                                "IKEA was founded in 1943 by Ingvar Kamprad in Almhult, Sweden. In fiscal "
                                "year 2009 IKEA had record sales.</p></body></html>",
    "https://mirror.net/ikea-history": "<html><body><p>History: the company was founded in 1943 in "
                                       "Sweden.</p></body></html>",
    "https://spam.biz/ikea": "<html><body><p>Buy cheap furniture now!</p></body></html>",
}


class FakeWeb:
    def __init__(self):
        self.calls = {"search": 0, "fetch": 0, "search_net": 0, "fetch_net": 0}

    def search(self, q):
        self.calls["search"] += 1
        return [{"url": u, "title": u.rsplit("/", 1)[-1], "snippet": "IKEA founded" if "spam" not in u else "",
                 "domain": u.split("/")[2]} for u in ("https://spam.biz/ikea", "https://example.org/ikea",
                                                      "https://mirror.net/ikea-history")]

    def fetch(self, url):
        self.calls["fetch"] += 1
        return {"url": url, "status": 200, "html": PAGES.get(url, "")}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "store.sqlite")
    yield s
    s.close()


# ── actions ──────────────────────────────────────────────────────────────
def test_validate_masks_illegal_actions():
    with pytest.raises(InvalidDecision):
        validate(Decision("ANSWER"), "start", {})
    with pytest.raises(InvalidDecision):
        validate(Decision("OPEN", k=5), "results", {"results": 3})
    d = validate(Decision("OPEN", k=1, p=7.0), "results", {"results": 3})
    assert d.p == 1.0


# ── candidates ───────────────────────────────────────────────────────────
def test_year_ranking_prefers_attribute_sentence():
    text = ("IKEA is a company. IKEA was founded in 1943 by Ingvar Kamprad. "
            "In fiscal year 2009 IKEA grew. IKEA opened stores in 2011.")
    c = span_candidates("What year was IKEA founded?", text, "year")
    assert c[0]["value"] == "1943"
    assert c[0]["margin"] > 0


def test_number_units_and_entities():
    assert question_unit("How tall is Mount Everest in metres?") == "m"
    text = "Mount Everest is 8,849 m (29,032 ft) tall. It was first climbed in 1953."
    c = span_candidates("How tall is Mount Everest in metres?", text, "number")
    assert c[0]["value"].startswith("8,849")
    e = span_candidates("What is the capital of Canada?", "Canada's capital is Ottawa, in Ontario.", "entity")
    assert "Ottawa" in [x["value"] for x in e]
    assert "Canada's" not in [x["value"] for x in e]


# ── metrics ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("ans,gold,t,ok", [
    ("1943", ["1943"], "year", True), ("28 July 1943", ["1943"], "year", True),
    ("8,848.86 m", ["8849"], "number", True), ("29,032 ft", ["8849"], "number", False),
    ("Miyazaki", ["Hayao Miyazaki"], "entity", True), ("Brasilia", ["Brasília"], "entity", True),
    ("Sun", ["Jupiter"], "entity", False), (None, ["x"], "entity", False),
])
def test_is_correct(ans, gold, t, ok):
    assert is_correct(ans, gold, t) is ok


def test_ece_and_summary():
    assert ece([1.0, 1.0], [True, True]) == 0.0
    assert abs(ece([0.9, 0.9], [False, False]) - 0.9) < 1e-9
    s = summarize([{"family": "fact", "answered": True, "correct": True, "p": 0.9, "steps": 4, "web_calls": 2},
                   {"family": "fact", "answered": False, "correct": False, "p": 0.1, "steps": 3, "web_calls": 2}])
    assert s["success"] == 0.5 and s["abstain_rate"] == 0.5 and s["wrong_when_answered"] == 0.0


# ── store ────────────────────────────────────────────────────────────────
def test_store_ttl_prior_watch(tmp_path):
    now = [1000.0]
    s = Store(tmp_path / "s.sqlite", clock=lambda: now[0])
    assert ttl_for("price", "What is the price of X?") == 1
    assert ttl_for("founded", "What year was X founded?") is None
    s.add_fact(question="What is the price of X?", value="10", answer_type="number", source_url="u",
               domain="d.com", context="c", p=0.8, entity="X", attribute="price")
    assert s.lookup("What is the price of X?", "X", "price")
    now[0] += 2 * DAY
    assert s.lookup("What is the price of X?", "X", "price") == []  # stale after 1-day TTL
    assert s.domain_prior("d.com") == 0.5
    s.update_domain("d.com", True)
    assert s.domain_prior("d.com") > 0.5
    wid = s.add_watch("What is the price of X?", "number", {"op": "lt", "value": 5})
    s.update_watch(wid, "10")
    assert s.watches()[0]["last_value"] == "10"
    s.close()


# ── harness ──────────────────────────────────────────────────────────────
def test_harness_heuristic_end_to_end(store):
    h = Harness(HeuristicPolicy(), FakeWeb(), store)
    res = h.run_fact("What year was IKEA founded?", "year", entity="IKEA", attribute="founded")
    assert res["answered"] and res["value"] == "1943", res["trajectory"]
    assert "example.org" in res["source_url"] or "mirror.net" in res["source_url"]
    assert store.n_facts() == 1
    assert "1943" in render_card("What year was IKEA founded?", res)
    # second ask is served from the store (compounding), zero web calls
    web = FakeWeb()
    res2 = Harness(HeuristicPolicy(), web, store).run_fact("What year was IKEA founded?", "year",
                                                           entity="IKEA", attribute="founded")
    assert res2["from_store"] and res2["value"] == "1943"
    assert web.calls["search"] + web.calls["fetch"] == 0


class ScriptedLLM(LLMPolicy):
    """Replays fixed completions: tests prompt rendering + JSON parsing + fallback."""
    name = "scripted"

    def __init__(self, replies):
        super().__init__()
        self.replies = list(replies)
        self.prompts = []

    def _complete(self, system, user):
        self.prompts.append(user)
        return self.replies.pop(0)


def test_llm_policy_parse_and_fallback(store):
    pol = ScriptedLLM(['{"action": "SEARCH", "k": null, "p": 0.5}',
                       'Sure! {"action":"OPEN","k":1,"p":0.6}',
                       '{"action": "EXTRACT", "k": 0, "p": 0.8}',
                       '{"action": "ANSWER", "k": null, "p": 0.85}'])
    res = Harness(pol, FakeWeb(), store).run_fact("What year was IKEA founded?", "year")
    assert res["value"] == "1943" and res["invalid"] == 0
    assert "RESULTS:" in pol.prompts[1] and "SPANS" in pol.prompts[2]
    garbage = ScriptedLLM(["I think the answer is 1943", '{"action": "DANCE"}'])
    res = Harness(garbage, FakeWeb(), store, use_store=False).run_fact("What year was IKEA founded?", "year")
    assert not res["answered"] and res["invalid"] == 2


def test_parse_decision():
    d = parse_decision('```json\n{"action": "open", "k": "2", "p": "0.7"}\n```')
    assert (d.action, d.k, d.p) == ("OPEN", 2, 0.7)
    assert parse_decision("no json").action == "INVALID"


def test_render_observation_hides_unpointed_lists():
    obs = {"question": "q", "answer_type": "year", "allowed": ["LOOKUP", "SEARCH"], "pointer": {},
           "lists": {"results": [{"title": "t", "domain": "d", "snippet": "s"}], "spans": [], "store": []},
           "held": None, "verified": None, "history": [], "opened": set()}
    assert "RESULTS" not in render_observation(obs)


# ── wikirace on a synthetic dump ─────────────────────────────────────────
def _page(title, text, redirect=None):
    red = f'<redirect title="{redirect}" />' if redirect else ""
    return (f"<page><title>{title}</title><ns>0</ns>{red}<revision><text>{text}</text></revision></page>")


def _mini_dump(path):
    # chain A -> B -> C -> D plus distractors, one redirect, one File: link
    lead = "is an example article with enough words to count as a lead paragraph here."
    pages = [
        _page("A", f"'''A''' {lead} [[B]] [[X]] [[File:pic.jpg]]"),
        _page("B", f"'''B''' {lead} [[C]] [[Y]] [[A]]"),
        _page("C", f"'''C''' {lead} [[Dee]] [[Z]]"),
        _page("D", f"'''D''' {lead} [[A]]"),
        _page("Dee", "#REDIRECT [[D]]", redirect="D"),
        _page("X", f"'''X''' {lead} [[Y]] [[B]]"),
        _page("Y", f"'''Y''' {lead} [[Z]] [[C]]"),
        _page("Z", f"'''Z''' {lead} [[X]] [[D]]"),
    ]
    xml = '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">' + "".join(pages) + "</mediawiki>"
    with bz2.open(path, "wt", encoding="utf-8") as f:
        f.write(xml)


def test_wikirace_pipeline(tmp_path):
    from src.planck3 import wikirace as wr
    dump = tmp_path / "mini.xml.bz2"
    _mini_dump(dump)
    wr.build_graph(dump, tmp_path)
    g = wr.Graph(tmp_path)
    assert g.n == 7  # redirect page dropped
    c = g.titles.index("C")
    assert g.titles.index("D") in g.out(c).tolist()  # [[Dee]] resolved through the redirect
    assert "pic" not in " ".join(g.titles)
    # every node has in-degree >= 1 here; lower the pool bar via small targets
    wr.make_tasks(tmp_path, n_targets=7, starts_per_target=2, offpath_per_target=2, seed=0,
                  min_dist=1, max_dist=4, min_indeg=1)
    from src.planck3.util import read_jsonl
    steps = [s for sp in ("train", "val", "test") for s in read_jsonl(tmp_path / f"steps_{sp}.jsonl")]
    assert steps and all(s["gold"] for s in steps if s["dist"] >= 1)
    for s in steps:  # every gold link is exactly one step closer
        assert all(v in g.out(s["node"]).tolist() for v in s["gold"])


def test_wikirace_train_and_rollout(tmp_path):
    import torch
    from src.planck3 import wikirace as wr
    from src.planck3.heads import PointerHead, multi_positive_nll
    head = PointerHead(16, hidden=8)
    e_t, e_n, e_c = torch.randn(2, 16), torch.randn(2, 16), torch.randn(2, 5, 16)
    gm = torch.zeros(2, 5, dtype=torch.bool)
    gm[:, 0] = True
    cm = torch.ones(2, 5, dtype=torch.bool)
    loss = multi_positive_nll(head(e_t, e_n, e_c), gm, cm)
    loss.backward()
    assert torch.isfinite(loss)
    dump = tmp_path / "mini.xml.bz2"
    _mini_dump(dump)
    wr.build_graph(dump, tmp_path)
    g = wr.Graph(tmp_path)
    E = np.eye(g.n, dtype=np.float32)
    ok, steps, _ = wr.rollout(g, wr.EmbedWR(E), g.titles.index("C"), g.titles.index("D"), max_steps=4)
    assert ok and steps == 1  # target is a direct link and cosine(target, target) = 1


def test_planck_encoder_on_tiny_checkpoint(tmp_path):
    """The real Planck 1.3 checkpoint lives on the GPU box; exercise the same code path on a tiny one."""
    import sentencepiece as spm
    import torch
    from src.planck3.encoders import PlanckEncoder
    from src.sgs_lm import SGSLanguageModel
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["the quick brown fox jumps over the lazy dog",
                                 "wikipedia is a free online encyclopedia",
                                 "ikea was founded in sweden by ingvar kamprad"] * 30), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / "tok"), vocab_size=60,
                                   model_type="unigram", hard_vocab_limit=False)
    model = SGSLanguageModel(vocab_size=60, d_s=8, d_f=16, n_passes=2, n_heads=2, max_len=32)
    torch.save({"model": model.state_dict()}, tmp_path / "tiny.pt")
    enc = PlanckEncoder(str(tmp_path / "tiny.pt"), str(tmp_path / "tok.model"), device="cpu")
    E = enc.encode(["IKEA. A Swedish company.", "Dog", "a much longer sentence about the lazy brown fox"],
                   batch_size=2)
    assert E.shape == (3, 16)
    assert np.allclose(np.linalg.norm(E.astype(np.float32), axis=1), 1.0, atol=1e-2)
    # right-padding must not change a sequence's embedding (causal model + masked mean)
    alone = enc.encode(["Dog"], batch_size=1)
    assert np.allclose(alone[0].astype(np.float32), E[1].astype(np.float32), atol=2e-2)


# ── chat surface ─────────────────────────────────────────────────────────
def test_store_fuzzy_lookup_never_swaps_subject(store):
    store.add_fact(question="What is the capital of Canada?", value="Ottawa", answer_type="entity",
                   source_url="u", domain="d", context="c", p=0.9)
    assert store.lookup("What is the capital of Canada?")          # same question hits
    assert store.lookup("what is the capital of canada")           # case/punctuation don't matter
    assert store.lookup("What is the capital of Brazil?") == []    # different subject never hits


@pytest.mark.parametrize("q,t", [
    ("When was IKEA founded?", "year"), ("In what year did the Berlin Wall fall?", "year"),
    ("How tall is the Burj Khalifa?", "number"), ("What is the atomic number of gold?", "number"),
    ("Who founded SpaceX?", "entity"), ("Where is Nintendo headquartered?", "entity"),
    ("What is the capital of Canada?", "entity"), ("On what date did Apollo 11 land?", "date"),
])
def test_infer_answer_type(q, t):
    from src.planck3.chat import infer_answer_type
    assert infer_answer_type(q) == t


def test_followups():
    from src.planck3.chat import Turn, main_entity, resolve_followup
    assert main_entity("When was IKEA founded?") == "IKEA"
    assert main_entity("How tall is the Burj Khalifa in metres?") == "Burj Khalifa"
    prev = Turn("When was IKEA founded?", "When was IKEA founded?", "year", "IKEA", "NEW")
    assert resolve_followup("And H&M?", prev)[:2] == ("When was H&M founded?", "SWAP_ENTITY")
    assert resolve_followup("what about Zara", prev)[:2] == ("When was Zara founded?", "SWAP_ENTITY")
    prev = Turn("Who founded SpaceX?", "Who founded SpaceX?", "entity", "SpaceX", "NEW")
    assert resolve_followup("When was it founded?", prev)[:2] == ("When was SpaceX founded?", "PRONOUN")
    assert resolve_followup("Who painted the Mona Lisa?", prev)[1] == "NEW"


def test_chat_session_and_server(tmp_path):
    import json as _json
    import threading
    import urllib.request
    from src.planck3.chat import ChatSession, chat_answer
    from src.planck3.serve import make_server
    st = Store(tmp_path / "chat.sqlite")
    sess = ChatSession(Harness(HeuristicPolicy(), FakeWeb(), st))
    t = sess.ask("When was IKEA founded?")
    assert t.answer_type == "year" and t.result["value"] == "1943"
    assert "**1943**" in chat_answer(t) and "Source:" in chat_answer(t)
    srv = make_server(lambda: Harness(HeuristicPolicy(), FakeWeb(), st), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    assert b"Planck 3.0" in urllib.request.urlopen(url + "/").read()
    req = urllib.request.Request(url + "/api/chat", data=_json.dumps({"message": "When was IKEA founded?",
                                                                      "session": "s"}).encode())
    r = _json.loads(urllib.request.urlopen(req).read())
    assert "1943" in r["reply"] and r["answer_type"] == "year"
    srv.shutdown()
    st.close()


def test_cost_model():
    from src.planck3.cost import load_prices, task_cost
    prices = load_prices()
    llm = task_cost(prices, "llm", {"input_tokens": 1_000_000, "output_tokens": 0}, 1, "searxng", [])
    assert llm["llm_usd"] == prices["llm_per_mtok"][prices["llm_equivalent"]]["input"]
    local = task_cost(prices, "local", {}, 2, "searxng", [1.0, 2.0])
    assert local["llm_usd"] == 0 and local["total_usd"] < 1e-6


# ── depth + the growing personal knowledge graph ─────────────────────────
def test_rank_passages_dedups_mirrors_and_skips_noise():
    from src.planck3.candidates import rank_passages
    docs = [{"url": "a", "domain": "a.org", "trust": 0.8,
             "text": "IKEA was founded in 1943 by Ingvar Kamprad in Sweden. It sells furniture."},
            {"url": "b", "domain": "mirror.net", "trust": 0.4,
             "text": "IKEA was founded in 1943 by Ingvar Kamprad in Sweden! It sells furniture."},
            {"url": "c", "domain": "c.org", "trust": 0.5, "text": "This is a page that is about nothing and it is long."}]
    ps = rank_passages("When was IKEA founded?", docs)
    assert ps and ps[0]["url"] == "a"                      # most trusted copy kept
    assert all(p["url"] != "b" for p in ps)               # its mirror dropped
    assert all(p["url"] != "c" for p in ps)               # stopword-only overlap is not a hit


def test_depth_pack_graph_and_local_recall(store):
    h = Harness(HeuristicPolicy(), FakeWeb(), store)
    res = h.run_fact("What year was IKEA founded?", "year", entity="IKEA", attribute="founded")
    d = res["depth"]
    assert d["passages"] and d["sources"] and d["n_pages_read"] >= 1
    assert any("1943" in p["text"] for p in d["passages"])
    g = store.graph_stats()
    assert g["passages"] >= 1 and g["queries"] == 1 and g["entities"] >= 1
    assert "ingvar kamprad" in [e for e, _ in store.adjacent("IKEA")]
    # the same question later is answered from memory AND its evidence comes back with zero web calls
    web = FakeWeb()
    res2 = Harness(HeuristicPolicy(), web, store).run_fact("What year was IKEA founded?", "year",
                                                           entity="IKEA", attribute="founded")
    assert res2["from_store"] and res2["depth"]["passages"]
    assert web.calls["search"] + web.calls["fetch"] == 0


def test_digest_and_feedback(store):
    from src.planck3.digest import build_digest, render_digest, trusted_domains
    assert "Nothing yet" in render_digest(build_digest(store))
    Harness(HeuristicPolicy(), FakeWeb(), store).run_fact("What year was IKEA founded?", "year", entity="IKEA")
    store.feedback("source", "example.org", 1)             # explicit trust beats implicit
    assert "example.org" in trusted_domains(store)
    d = build_digest(store)
    assert d["interests"][0]["entity"] == "ikea"
    assert any(a["entity"] == "ingvar kamprad" for a in d["interests"][0]["adjacent"])
    assert "For you" in render_digest(d)
    store.feedback("source", "spam.biz", -1)
    assert store.domain_prior("spam.biz") < 0.5


def test_graph_nodes_are_names_not_adjectives():
    from src.planck3.candidates import passage_entities
    ents = passage_entities("SpaceX is an American company founded in February 2002 by Elon Musk. "
                            "The CEO of SpaceX lives in Texas.", exclude="SpaceX")
    assert "Elon Musk" in ents and "Texas" in ents
    assert not {"American", "February", "CEO of SpaceX"} & set(ents)


def test_closed_book_comparator(tmp_path, monkeypatch):
    """--closed-book scores the base-chat rival with the same rules and cost model."""
    import importlib.util
    import json as _json
    spec = importlib.util.spec_from_file_location("p3cli", Path(__file__).resolve().parent.parent / "scripts" / "planck3.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    fake = ScriptedLLM(["1943", "IKEA was founded in 1943.", "Probably 1950"])
    monkeypatch.setattr(cli, "make_policy", lambda args: fake)
    tasks = tmp_path / "t.json"
    tasks.write_text(_json.dumps({"tasks": [
        {"id": "f", "family": "fact", "question": "What year was IKEA founded?", "answer_type": "year", "gold": ["1943"]},
        {"id": "c", "family": "chat", "turns": [
            {"user": "When was IKEA founded?", "answer_type": "year", "gold": ["1943"]},
            {"user": "And H&M?", "answer_type": "year", "gold": ["1947"]}]}]}), encoding="utf-8")
    sys.argv = ["planck3.py", "g0", "--policy", "gemma", "--closed-book", "--tasks", str(tasks),
                "--out", str(tmp_path / "run")]
    cli.main()
    s = _json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert s["mode"] == "closed_book" and s["n"] == 2 and abs(s["success"] - 0.5) < 1e-9
    assert "User: When was IKEA founded?" in fake.prompts[-1]   # chat turns carry the conversation
