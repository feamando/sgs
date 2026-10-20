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
    assert s.domain_prior("d.com") == 0.5                       # unscored domain: 5/10
    s.update_domain("d.com", True)
    assert s.domain_prior("d.com") == 0.5                       # usage stats no longer move trust (round 5)
    s.feedback("source", "d.com", 1)
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
    store.feedback("source", "example.org", 1)             # one click: 5 -> personal 7 -> effective 5.6
    assert "example.org" not in trusted_domains(store)     # the 30% cap: one click cannot make a source trusted
    store.feedback("source", "example.org", 1)
    store.feedback("source", "example.org", 1)             # personal 10 -> effective 6.5
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


# ── doctor ───────────────────────────────────────────────────────────────
def test_doctor_catches_tokenizer_mismatch_and_measures(tmp_path):
    import sentencepiece as spm
    import torch
    from src.planck3.doctor import FAIL, OK, Doctor
    from src.sgs_lm import SGSLanguageModel
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join(["the quick brown fox jumps over the lazy dog", "ikea was founded in sweden"] * 40),
                      encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / "tok"), vocab_size=50,
                                   model_type="unigram", hard_vocab_limit=False)
    n = spm.SentencePieceProcessor(model_file=str(tmp_path / "tok.model")).get_piece_size()
    for vocab, expect in ((n, OK), (n + 7, FAIL)):
        m = SGSLanguageModel(vocab_size=vocab, d_s=8, d_f=16, n_passes=2, n_heads=2, max_len=32)
        torch.save({"model": m.state_dict()}, tmp_path / "m.pt")
        d = Doctor()
        d.planck(str(tmp_path / "m.pt"), str(tmp_path / "tok.model"), deep=(expect == OK))
        row = next(r for r in d.rows if r["check"] == "planck ckpt vs tokenizer")
        assert row["status"] == expect
    assert d.measured == {} and "planck_embed_s_per_text" not in d.measured  # FAIL case did not measure
    assert any(stage.startswith("g1 planck embed") for stage, _ in d.eta())


def test_doctor_probe_observations_are_valid_decision_points():
    from src.planck3.actions import validate
    from src.planck3.doctor import _probe_observations
    pol = HeuristicPolicy()
    for obs in _probe_observations():
        d = pol.decide(obs)
        validate(d, obs["phase"], {k: len(v) for k, v in obs["lists"].items()})


# ── snippet-first + confidence gate (2026-10-07) ─────────────────────────
class SnippetWeb(FakeWeb):
    """Snippets that already carry the answer, from two domains about the subject, plus a distractor."""
    def search(self, q):
        self.calls["search"] += 1
        return [{"url": "https://other.org/ikea-family", "title": "IKEA Family magazine", "domain": "other.org",
                 "snippet": "The IKEA Family magazine was launched in 1995."},
                {"url": "https://example.org/ikea", "title": "IKEA", "domain": "example.org",
                 "snippet": "IKEA was founded in 1943 by Ingvar Kamprad."},
                {"url": "https://mirror.net/ikea-history", "title": "IKEA history", "domain": "mirror.net",
                 "snippet": "IKEA history: the company was founded in 1943 in Sweden."}]


def test_snippet_first_answers_without_fetching_and_verifies_free(store):
    pol = ScriptedLLM(['{"action": "SEARCH", "k": null, "p": 0.5}',
                       '{"action": "EXTRACT", "k": 0, "p": 0.8}',
                       '{"action": "VERIFY", "k": null, "p": 0.8}',
                       '{"action": "ANSWER", "k": null, "p": 0.9}'])
    web = SnippetWeb()
    res = Harness(pol, web, store).run_fact("What year was IKEA founded?", "year")
    assert res["value"] == "1943" and res["from_snippet"] and res["verified"]   # 2 domains agree: free VERIFY
    assert res["answer_fetch_calls"] == 0                                      # nothing fetched to answer
    assert "search result snippets" in pol.prompts[1] and "2 domains" in pol.prompts[1]


def test_snippet_aboutness_beats_a_page_about_something_else():
    from src.planck3.candidates import snippet_candidates
    results = [{"url": "https://a.org/x", "title": "Netflix Animation", "snippet": "Netflix Animation was founded in 2018."},
               {"url": "https://b.org/y", "title": "Netflix", "snippet": "Netflix was founded in 1997 in California."}]
    c = snippet_candidates("What year was Netflix founded?", results, "year")
    assert c[0]["value"] == "1997" and c[0]["about"] == 1.0


def test_confidence_gate_turns_a_guess_into_abstain(store):
    pol = ScriptedLLM(['{"action": "SEARCH", "k": null, "p": 0.5}',
                       '{"action": "EXTRACT", "k": 0, "p": 0.1}',
                       '{"action": "ANSWER", "k": null, "p": 0.05}'])
    res = Harness(pol, SnippetWeb(), store, use_store=False).run_fact("What year was IKEA founded?", "year")
    assert not res["answered"] and res["reason"] == "low_confidence" and res["gated_value"] == "1943"
    off = ScriptedLLM(['{"action": "SEARCH", "k": null, "p": 0.5}', '{"action": "EXTRACT", "k": 0, "p": 0.1}',
                       '{"action": "ANSWER", "k": null, "p": 0.05}'])
    res = Harness(off, SnippetWeb(), store, use_store=False, answer_threshold=0).run_fact("What year was IKEA founded?", "year")
    assert res["answered"]                                                      # threshold 0 disables the gate


def test_pages_first_still_available(store):
    h = Harness(HeuristicPolicy(), SnippetWeb(), store, snippet_first=False)
    res = h.run_fact("What year was IKEA founded?", "year", entity="IKEA")
    assert not res["from_snippet"] and res["answer_fetch_calls"] >= 1


# ── G1 follow-up: distances, rank objective, paired stats, seeds ─────────
def test_cand_dist_labels_and_rank_loss(tmp_path):
    import torch
    from src.planck3 import wikirace as wr
    from src.planck3.util import read_jsonl
    dump = tmp_path / "mini.xml.bz2"
    _mini_dump(dump)
    wr.build_graph(dump, tmp_path)
    g = wr.Graph(tmp_path)
    wr.make_tasks(tmp_path, n_targets=7, starts_per_target=2, offpath_per_target=2, seed=0,
                  min_dist=1, max_dist=4, min_indeg=1)
    steps = [s for sp in ("train", "val", "test") for s in read_jsonl(tmp_path / f"steps_{sp}.jsonl")]
    for s in steps:
        nb = g.out(s["node"]).tolist()
        assert len(s["cand_dist"]) == len(nb)
        assert set(s["gold"]) == {v for v, d in zip(nb, s["cand_dist"]) if d == s["dist"] - 1}
    dist = torch.tensor([[0.0, 1.0, 3.0, -1.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    good = wr.rank_loss(torch.tensor([[3.0, 2.0, 0.0, -2.0]]), dist, mask)
    bad = wr.rank_loss(torch.tensor([[-2.0, 0.0, 2.0, 3.0]]), dist, mask)
    assert good < bad


def test_mcnemar_exact():
    from src.planck3.wikirace import mcnemar
    r = mcnemar([True] * 10 + [False] * 10, [False] * 10 + [False] * 10)
    assert r["a_only"] == 10 and r["b_only"] == 0 and r["p"] < 0.01
    assert mcnemar([True, False], [True, False])["p"] == 1.0


def test_eval_reuses_teacher_cache_and_reports_paired(tmp_path, monkeypatch):
    import json as _json
    from src.planck3 import wikirace as wr
    from src.planck3.encoders import HashEncoder
    dump = tmp_path / "mini.xml.bz2"
    _mini_dump(dump)
    wr.build_graph(dump, tmp_path)
    g = wr.Graph(tmp_path)
    wr.make_tasks(tmp_path, n_targets=7, starts_per_target=2, offpath_per_target=1, seed=0,
                  min_dist=1, max_dist=4, min_indeg=1)
    import numpy as np
    np.save(tmp_path / "emb_hash.npy", HashEncoder(64).encode([g.text(u) for u in range(g.n)]))
    pairs = [_json.loads(l) for l in open(tmp_path / "pairs_test.jsonl", encoding="utf-8")]
    tdir = tmp_path / "teacher"
    tdir.mkdir()
    with open(tdir / "pairs_gemma.jsonl", "w", encoding="utf-8") as f:   # pretend the teacher already ran
        for i, p in enumerate(pairs):
            f.write(_json.dumps({"i": i, "start": p["start"], "target": p["target"], "dist": p["dist"],
                                 "ok": i % 2 == 0, "steps": p["dist"], "ms": 200.0}) + "\n")
    monkeypatch.setattr(wr, "GemmaWR", lambda *a, **k: (_ for _ in ()).throw(AssertionError("teacher must not run")))
    wr.train(tmp_path, "hash", tmp_path / "results" / "g1_head_hash_s0", epochs=1, lr=1e-3, seed=0)
    heads = {"hash": tmp_path / "results" / "g1_head_hash_s0" / "head.pt"}
    out = tmp_path / "eval"
    wr.evaluate(tmp_path, ["random", "lexical", "heads", "gemma"], out, 0, "unused", len(pairs), heads,
                None, None, teacher_dir=tdir)
    s = _json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["results"]["gemma"]["cached"] is True
    assert "head:hash vs gemma" in s["paired"] and "head:hash vs lexical" in s["paired"]
    assert (out / "pairs_head_hash.jsonl").exists()


def test_aggregate_over_seeds(tmp_path):
    import json as _json
    from src.planck3 import wikirace as wr
    for sd, (pl, hs, ratio) in enumerate([(0.21, 0.15, 0.70), (0.23, 0.14, 0.77), (0.22, 0.16, 0.73)]):
        d = tmp_path / f"g1_eval_s{sd}"
        d.mkdir()
        (d / "summary.json").write_text(_json.dumps({
            "results": {"head:planck": {"rollout_success": pl}, "head:hash": {"rollout_success": hs},
                        "gemma": {"rollout_success": 0.30}},
            "paired": {"head:planck vs gemma": {"ratio": ratio}}}), encoding="utf-8")
    wr.aggregate(tmp_path, [0, 1, 2], "", tmp_path / "agg")
    s = _json.loads((tmp_path / "agg" / "summary.json").read_text(encoding="utf-8"))
    assert abs(s["policies"]["head:planck"]["mean"] - 0.22) < 1e-9 and s["policies"]["head:planck"]["ci95"]
    assert s["gate"]["verdict"] == "FAIL" and abs(s["paired_ratio_vs_teacher"]["head:planck"]["mean"] - 0.7333) < 1e-3


# ── G2: learn the decisions from known answers ───────────────────────────
def test_g2_collect_train_and_policy(tmp_path, monkeypatch):
    import json as _json
    from src.planck3 import g2
    tasks = [{"id": f"t{i}", "family": "fact", "question": "What year was IKEA founded?", "answer_type": "year",
              "gold": ["1943"]} for i in range(12)]
    pts = tmp_path / "points.jsonl"
    g2.collect(tasks, SnippetWeb(), pts)
    rows = [_json.loads(l) for l in open(pts, encoding="utf-8")]
    assert rows and {r["source"] for r in rows} == {"snippet", "page"}
    snip = next(r for r in rows if r["source"] == "snippet")
    assert any(snip["labels"]) and snip["cands"][snip["labels"].index(True)]["value"] == "1943"
    g2.collect(tasks, SnippetWeb(), pts)                                  # resumable: nothing re-collected
    assert len([_json.loads(l) for l in open(pts, encoding="utf-8")]) == len(rows)
    emb = tmp_path / "g2"
    g2.embed(pts, "hash", emb)
    g2.train(pts, emb, "hash", tmp_path / "head", epochs=2, seed=0)
    log = _json.loads((tmp_path / "head" / "train_log.json").read_text(encoding="utf-8"))
    assert 0 <= log["best_val_top1"] <= 1 and log["temperature"] > 0
    pol = g2.PlanckPolicy(str(tmp_path / "head" / "head.pt"))
    store = Store(tmp_path / "s.sqlite")
    res = Harness(pol, SnippetWeb(), store).run_fact("What year was IKEA founded?", "year")
    assert res["steps"] >= 2 and res["invalid"] == 0                       # a legal typed trajectory
    feats = g2.features({"value": "IKEA", "score": 1, "margin": 0.1}, "What year was IKEA founded?", "year", 0, 3, True)
    assert len(feats) == g2.N_FEAT and feats[6] == 1.0                     # subject echo flagged
    store.close()


def test_benchgen_task_shape_offline(monkeypatch):
    from src.planck3 import benchgen as bg
    rows = [{"item": "http://www.wikidata.org/entity/Q1", "itemLabel": "2026 Test Cup final", "ans": "x",
             "ansLabel": "Team A", "aliases": "A FC|Team A Club"}]
    monkeypatch.setattr(bg, "sparql", lambda q: rows if "P1346" in q and "P585" in q else [])
    tasks = bg.generate("bench")
    t = tasks[0]
    assert t["question"] == "Who won the 2026 Test Cup final?" and t["regime"] == "fresh"
    assert t["gold"][:2] == ["Team A", "A FC"] and t["qid"] == "Q1" and t["gold_as_of"]
    assert bg._clean_desc("American footballer (born 1987)") is None   # answer-leaking description dropped


# ── search robustness (round 3 failure: SearXNG blocked, HTTP 200 + empty results) ──
def _web(tmp_path, monkeypatch, searx, wiki):
    from src.planck3 import web as W
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    w = W.Web(W.WebCache(tmp_path / "cache"), search_backend="searxng")
    monkeypatch.setattr(w, "_search_searxng", searx)
    monkeypatch.setattr(w, "_search_wikipedia", wiki)
    return w


def test_search_falls_back_to_wikipedia_when_searxng_is_empty(tmp_path, monkeypatch):
    hit = [{"url": "https://en.wikipedia.org/wiki/IKEA", "title": "IKEA", "snippet": "founded 1943", "domain": "en.wikipedia.org"}]
    w = _web(tmp_path, monkeypatch, lambda q: [], lambda q: hit)
    assert w.search("What year was IKEA founded?") == hit
    h = w.search_health()
    assert h["fallback_rate"] == 1.0 and h["final_empty_rate"] == 0.0 and h["valid"]
    assert w.search("What year was IKEA founded?") == hit   # served from the wikipedia cache entry


def test_search_circuit_breaker_stops_a_blind_run(tmp_path, monkeypatch):
    from src.planck3.web import MAX_CONSECUTIVE_EMPTY, SearchUnavailable
    w = _web(tmp_path, monkeypatch, lambda q: [], lambda q: [])
    for i in range(MAX_CONSECUTIVE_EMPTY - 1):
        assert w.search(f"q{i}") == []
    with pytest.raises(SearchUnavailable):
        w.search("one more")
    assert not w.search_health()["valid"]
    assert not list((tmp_path / "cache" / "search").glob("*.json"))   # failures are never cached


def test_searxng_alive_needs_real_results(tmp_path, monkeypatch):
    from src.planck3 import web as W
    w = W.Web(W.WebCache(tmp_path / "c"))

    class R:
        status_code = 200

        def __init__(self, results):
            self._r = results

        def json(self):
            return {"results": self._r}
    monkeypatch.setattr(w.session, "get", lambda *a, **k: R([]))
    assert not w.searxng_alive()                                   # up but blocked = not alive
    monkeypatch.setattr(w.session, "get", lambda *a, **k: R([{"url": "x"}]))
    assert w.searxng_alive()


# ── G2 v2: choice and answerability decoupled ────────────────────────────
def test_select_threshold_meets_precision_target():
    from src.planck3.g2 import select_threshold
    scores = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4]
    correct = [True, True, True, True, True, False, True, False]
    t = select_threshold(scores, correct, precision=0.9)
    sel = [c for s, c in zip(scores, correct) if s >= t]
    assert sum(sel) / len(sel) >= 0.9 and t == 0.7          # at 0.5 it would be 6/7 = 0.857 < 0.9
    assert select_threshold([0.9, 0.8], [False, False]) == 1.01   # never answer if nothing is precise enough


def _synthetic_points(n=160, seed=0):
    import random as _r
    rng = _r.Random(seed)
    pts = []
    for i in range(n):
        year = str(1900 + i % 90)
        right = rng.random() < 0.6                    # 40% of points have no right candidate
        vals = [str(1800 + rng.randrange(200)) for _ in range(4)]
        if right:
            vals[rng.randrange(4)] = year
        cands = [{"value": v, "context": f"The company was founded in {v}." if v == year else f"Something in {v}.",
                  "score": 1.2 if v == year else rng.random(), "margin": 0.1, "about": 1.0 if v == year else 0.3,
                  "domains": ["a.org"], "support": 1, "url": "https://a.org"} for v in vals]
        pts.append({"task_id": f"t{i}", "question": f"What year was Company{i} founded?", "answer_type": "year",
                    "source": "snippet", "cands": cands, "labels": [c["value"] == year for c in cands]})
    return pts


def test_g2v2_train_gate_and_policy_logs_argmax(tmp_path):
    import json as _json
    from src.planck3 import g2
    pts_path = tmp_path / "points_searxng.jsonl"
    pts_path.write_text("\n".join(_json.dumps(p) for p in _synthetic_points()), encoding="utf-8")
    g2.embed(pts_path, "hash", tmp_path)
    assert g2.emb_path(tmp_path, "hash", pts_path).exists()
    g2.train_v2(pts_path, tmp_path, "hash", tmp_path / "v2", epochs=3)
    log = _json.loads((tmp_path / "v2" / "train_log.json").read_text(encoding="utf-8"))
    assert log["version"] == 2 and 0 <= log["test_choice_acc"] <= 1 and log["tau"] > 0
    assert log["n_train_points"] > 0 and log["test_at_tau"]["coverage"] is not None
    pol = g2.load_policy(str(tmp_path / "v2" / "head.pt"))
    assert isinstance(pol, g2.PlanckPolicyV2)
    st = Store(tmp_path / "s.sqlite")
    res = Harness(pol, SnippetWeb(), st).run_fact("What year was IKEA founded?", "year")
    snip = [s for s in res["trajectory"] if s["phase"] == "results_snip"]
    assert snip and {"argmax", "gate", "top3", "tau"} <= set(snip[0]["meta"])   # every decision is diagnosable
    st.close()


def test_searxng_run_never_reads_old_wikipedia_cache(tmp_path, monkeypatch):
    """Round 3b bug: a 'SearXNG' run was served cached Wikipedia results without falling back."""
    from src.planck3 import web as W
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    cache = W.WebCache(tmp_path / "cache")
    wiki_hit = [{"url": "https://en.wikipedia.org/wiki/X", "title": "X", "snippet": "old", "domain": "en.wikipedia.org"}]
    cache.put("search", "wikipedia|q", {"query": "q", "backend": "wikipedia", "results": wiki_hit})
    searx_hit = [{"url": "https://example.org/x", "title": "X", "snippet": "new", "domain": "example.org"}]
    w = W.Web(cache, search_backend="searxng")
    monkeypatch.setattr(w, "_search_searxng", lambda q: searx_hit)
    assert w.search("q") == searx_hit and w.last_backend == "searxng"       # primary first, not the old cache
    h = w.search_health()
    assert h["served"] == {"searxng": 1} and h["primary_share"] == 1.0
    cache.put("search", "wikipedia|q2", {"query": "q2", "backend": "wikipedia", "results": wiki_hit})
    w2 = W.Web(cache, search_backend="searxng")
    monkeypatch.setattr(w2, "_search_searxng", lambda q: [])
    assert w2.search("q2") == wiki_hit and w2.last_backend == "wikipedia"  # the fallback cache only after failing
    assert w2.search_health()["primary_share"] == 0.0


# ── round 4: ddgs search, answer tiers + explanations, three-layer source trust ──
def test_ddgs_backend_maps_results_and_falls_back(tmp_path, monkeypatch):
    import types
    from src.planck3 import web as W
    exc = types.ModuleType("ddgs.exceptions")

    class DDGSException(Exception):
        pass

    class RatelimitException(DDGSException):
        pass
    exc.DDGSException, exc.RatelimitException = DDGSException, RatelimitException
    calls = []

    class DDGS:
        def __init__(self, **kw):
            pass

        def text(self, query, **kw):
            calls.append((query, kw["backend"]))
            if query == "blocked":
                raise DDGSException("No results found.")
            return [{"title": "IKEA", "href": "https://www.ikea.com/about", "body": "IKEA was founded in 1943."},
                    {"title": "dup", "href": "https://www.ikea.com/about", "body": "x"}]
    mod = types.ModuleType("ddgs")
    mod.DDGS, mod.exceptions = DDGS, exc
    monkeypatch.setitem(sys.modules, "ddgs", mod)
    monkeypatch.setitem(sys.modules, "ddgs.exceptions", exc)
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    w = W.Web(W.WebCache(tmp_path / "c"), search_backend="ddgs")
    r = w.search("When was IKEA founded?")
    assert r == [{"url": "https://www.ikea.com/about", "title": "IKEA", "snippet": "IKEA was founded in 1943.",
                  "domain": "ikea.com"}] and w.last_backend == "ddgs"
    assert calls[0][1] == W.DDGS_ENGINES and "google" not in W.DDGS_ENGINES
    hit = [{"url": "https://en.wikipedia.org/wiki/X", "title": "X", "snippet": "s", "domain": "en.wikipedia.org"}]
    monkeypatch.setattr(w, "_search_wikipedia", lambda q: hit)
    assert w.search("blocked") == hit and w.last_backend == "wikipedia"      # empty -> retry -> Wikipedia
    h = w.search_health()
    assert h["served"] == {"ddgs": 1, "wikipedia": 1} and h["primary_share"] == 0.5
    assert w.ddgs_alive()
    assert w.search_site("ikea.com", "founded") and not W.Web(W.WebCache(tmp_path / "d"), search_backend="wikipedia").search_site("ikea.com", "q")


def test_measured_trust_table_is_audit_only(tmp_path):
    from src.planck3.trust import SYSTEM_MAX_STRENGTH, build_system
    import json as _json
    pts = tmp_path / "points_ddgs.jsonl"
    rows = []
    for i in range(30):
        rows.append({"source": "snippet", "cands": [{"value": "A", "domains": ["good.org"]},
                                                    {"value": "B", "domains": ["bad.com"]}], "labels": [True, False]})
    rows.append({"source": "snippet", "cands": [{"value": "Z", "domains": ["x.org"]}], "labels": [False]})  # nothing right: skipped
    rows += [{"source": "page", "cands": [{"value": "C", "domains": ["page.net"]}], "labels": [True]}] * 5      # pages: skipped
    pts.write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")
    table = build_system([pts], tmp_path / "trust.json")
    good, bad = table["domains"]["good.org"], table["domains"]["bad.com"]
    assert good["rate"] == 1.0 and bad["rate"] == 0.0 and not {"x.org", "page.net"} & set(table["domains"])
    assert good["a"] + good["b"] <= SYSTEM_MAX_STRENGTH + 1e-6
    st = Store(tmp_path / "s.sqlite")
    assert st.domain_prior("good.org") == st.domain_prior("bad.com") == 0.5      # measured table never scores sources
    st.close()


def test_answer_tiers_and_explanation(store):
    from src.planck3.chat import Turn, chat_answer, render_why
    res = Harness(HeuristicPolicy(), SnippetWeb(), store, use_store=False).run_fact("What year was IKEA founded?", "year")
    ex = res["explain"]
    assert res["tier"] == "confident" and ex["value"] == "1943" and not ex["calibrated"]
    assert ex["steps"][0].startswith("Searched") and ex["steps"][-1].startswith("Answered")
    assert {s["role"] for s in ex["sources"]} >= {"answer", "agrees"}                     # 2nd snippet domain agrees
    assert {"system", "personal", "effective", "band", "family"} <= set(ex["sources"][0]["trust"])
    assert any("question's words" in k for k, _ in ex["why_value"])
    t = Turn("q", "What year was IKEA founded?", "year", "IKEA", "NEW", res)
    assert "**1943**" in chat_answer(t) and "agrees" in chat_answer(t) and "How I got this" in render_why(t)
    # a value the policy will not commit to is SHOWN, labelled low confidence
    pol = ScriptedLLM(['{"action": "SEARCH", "k": null, "p": 0.5}', '{"action": "EXTRACT", "k": 0, "p": 0.1}',
                       '{"action": "ANSWER", "k": null, "p": 0.05}'])
    low = Harness(pol, SnippetWeb(), store, use_store=False).run_fact("What year was IKEA founded?", "year")
    assert not low["answered"] and low["tier"] == "low_confidence" and low["guess"]["value"] == "1943"
    reply = chat_answer(Turn("q", "What year was IKEA founded?", "year", "IKEA", "NEW", low))
    assert reply.startswith("Low confidence in my results") and "**1943**" in reply and "5%" in reply

    class Empty(FakeWeb):
        def search(self, q):
            self.calls["search"] += 1
            return []
    none = Harness(HeuristicPolicy(), Empty(), store, use_store=False).run_fact("Who is the mayor of Atlantis?", "entity")
    assert none["tier"] == "none" and none["guess"] is None and none["explain"]["steps"][-1].startswith("Nothing")


def test_g2v2_explains_its_confidence_with_gate_weights(tmp_path):
    import json as _json
    from src.planck3 import g2
    pts_path = tmp_path / "points_ddgs.jsonl"
    pts_path.write_text("\n".join(_json.dumps(p) for p in _synthetic_points()), encoding="utf-8")
    g2.embed(pts_path, "hash", tmp_path)
    g2.train_v2(pts_path, tmp_path, "hash", tmp_path / "v2", epochs=2)
    pol = g2.load_policy(str(tmp_path / "v2" / "head.pt"))
    st = Store(tmp_path / "s.sqlite")
    res = Harness(pol, SnippetWeb(), st).run_fact("What year was IKEA founded?", "year")
    ex = res["explain"]
    assert ex["calibrated"] and ex["candidates"] and all("p" in c for c in ex["candidates"])
    assert ex["why_confidence"]["kind"].startswith("log-odds")
    names = [k for k, _ in ex["why_confidence"]["items"]]
    assert names[-1] == "baseline" and set(names[:-1]) <= set(g2.GATE_FEATURE_NAMES)
    assert res["tier"] in ("confident", "low_confidence")       # v2 always names its best candidate
    st.close()


def test_session_layer_more_from_here(tmp_path):
    class SiteWeb(SnippetWeb):
        backend = "ddgs"

        def search_site(self, domain, q):
            return [{"url": f"https://{domain}/deep", "title": "Deep dive", "domain": domain,
                     "snippet": "IKEA was founded in 1943; the first store opened in 1958."}]
    st = Store(tmp_path / "s.sqlite")
    h = Harness(HeuristicPolicy(), SiteWeb(), st)
    before = st.domain_prior("mirror.net")
    r = h.more_from("mirror.net", "What year was IKEA founded?")
    assert r["passages"] and h._trust("mirror.net")["combined"] == before         # retrieval only: trust unchanged
    res = h.run_fact("When did the first IKEA store open?", "year")
    assert any(s["session"] for s in res["depth"]["sources"])                      # later turns search it too
    fresh = Harness(HeuristicPolicy(), SiteWeb(), st)                              # "New chat": session gone
    assert fresh.session_more == []
    st.close()


def test_server_more_endpoint_and_explain(tmp_path):
    import json as _json
    import threading
    import urllib.request
    from src.planck3.serve import make_server
    st = Store(tmp_path / "chat.sqlite")
    srv = make_server(lambda: Harness(HeuristicPolicy(), SnippetWeb(), st), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, obj):
        return _json.loads(urllib.request.urlopen(urllib.request.Request(url + path, data=_json.dumps(obj).encode())).read())
    assert "error" in post("/api/more", {"session": "s", "domain": "example.org"})
    r = post("/api/chat", {"message": "When was IKEA founded?", "session": "s"})
    assert r["tier"] == "confident" and r["explain"]["steps"] and r["explain"]["sources"]
    m = post("/api/more", {"session": "s", "domain": "example.org"})
    assert m["trust"]["effective"] == 5.0 and m["passages"] is not None       # unscored, and unchanged by the button
    srv.shutdown()
    st.close()


def test_answer_tier_summary():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import importlib
    cli = importlib.import_module("planck3")
    recs = [{"tier": "confident", "shown_correct": True, "evidence_hit": True},
            {"tier": "confident", "shown_correct": False, "evidence_hit": True},
            {"tier": "low_confidence", "shown_correct": False, "evidence_hit": False},
            {"tier": "none", "shown_correct": False, "evidence_hit": True}]
    t = cli.answer_tiers(recs)
    assert t["confident"]["precision"] == 0.5 and t["confident"]["rate"] == 0.5
    assert t["low_confidence"]["precision"] == 0.0 and t["none"]["precision"] is None
    assert t["none"]["evidence_recall"] == 1.0


def test_snippet_publish_stamp_is_not_the_answer():
    from src.planck3.candidates import _unstamp, snippet_candidates
    assert _unstamp("Sep 13, 2026 \u00b7 Who won? Jane won.").startswith("Sep 13, 2026. Who won?")
    assert _unstamp("3 days ago \u00b7 Tadej won.") == "3 days ago. Tadej won."
    r = [{"url": "https://a.org/x", "title": "Final", "snippet": "Sep 13, 2026 \u00b7 Who won the girls' final? Jane Doe beat Mary Roe."}]
    vals = [c["value"] for c in snippet_candidates("Who won the girls' final?", r, "entity")]
    assert "Sep" not in vals and "Jane Doe" in vals


def test_env_file_loads_without_overriding(tmp_path, monkeypatch):
    import os
    from src.planck3.util import load_env
    env = tmp_path / ".env"
    env.write_text("# comment\nP3_TEST_A=one\nexport P3_TEST_B='two'\nP3_TEST_C=\nnot a line\n", encoding="utf-8")
    monkeypatch.delenv("P3_TEST_A", raising=False)
    monkeypatch.delenv("P3_TEST_C", raising=False)
    monkeypatch.setenv("P3_TEST_B", "already")
    assert load_env(env) == ["P3_TEST_A"]                         # names only, never values
    assert os.environ["P3_TEST_A"] == "one" and os.environ["P3_TEST_B"] == "already" and "P3_TEST_C" not in os.environ
    assert load_env(tmp_path / "missing.env") == []


def test_brave_backend_maps_results_and_needs_a_key(tmp_path, monkeypatch):
    from src.planck3 import web as W
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    w = W.Web(W.WebCache(tmp_path / "c"), search_backend="brave")
    seen = {}

    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"web": {"results": [{"url": "https://www.ikea.com/a", "title": "<b>IKEA</b>",
                                         "description": "Founded in <strong>1943</strong>."}]}}

    def get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, headers=headers)
        return R()
    monkeypatch.setattr(w.session, "get", get)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        w.search("When was IKEA founded?")                         # no key: a clear error, not a blind run
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    r = w.search("When was IKEA founded?")
    assert r == [{"url": "https://www.ikea.com/a", "title": "IKEA", "snippet": "Founded in 1943.", "domain": "ikea.com"}]
    assert seen["headers"]["X-Subscription-Token"] == "k" and seen["url"] == W.BRAVE_URL and w.last_backend == "brave"


# ── round 5: hand-scored registry, question types, read -> weigh -> write ─────
def test_registry_scores_families_and_the_30_percent_cap():
    from src.planck3.registry import CONFLICT_GAP, default_registry, effective, trust_record
    reg = default_registry()
    w, mirror, grok = reg.lookup("en.wikipedia.org"), reg.lookup("www.wikiwand.com"), reg.lookup("grokipedia.com")
    assert w["score"] == 9 and mirror["family"] == w["family"] == grok["family"] == "wikimedia" and grok["score"] == 4
    assert reg.lookup("de.wikipedia.org")["family"] == "wikimedia"                       # suffix rule keeps the family
    assert reg.lookup("meduza.io")["score"] == 10 and reg.lookup("dw.com")["score"] == 10
    assert reg.lookup("news.bbc.co.uk")["family"] == "bbc" and reg.lookup("bbc.com")["score"] == 7
    assert reg.lookup("cdc.gov")["score"] == 9 and reg.lookup("rt.com")["score"] <= 4
    a, b = reg.lookup("usa.gov"), reg.lookup("ed.gov")
    assert a["category"] == b["category"] == "us_gov" and a["family"] != b["family"]     # every .gov site is its own source
    u = reg.lookup("some-new-blog.example")
    assert u["score"] == 5 and not u["scored"]
    for s in (0, 4, 9, 10):
        for p in (0, 10):
            assert abs(effective(s, p) - s) <= 3.0 + 1e-9                               # personal moves a source <= 3 points
    t = trust_record("rt.com", personal=10.0)
    assert t["effective"] <= 0.7 * t["system"] + 3 + 1e-9 and t["conflict"] and t["band"] in ("low", "sceptical")
    assert not trust_record("bbc.com", personal=7 + CONFLICT_GAP - 1)["conflict"]


def test_question_types_and_penalties():
    from src.planck3.qtype import PENALTY, classify
    assert classify("Who won the 2026 US Open girls' singles?", 2026) == "news"
    assert classify("Who is the current CEO of Nokia?", 2026) == "news"
    assert classify("When was IKEA founded?", 2026) == "encyclopedic"
    assert classify("When was Port Moresby founded?", 2026) == "encyclopedic"           # 'Port' is a name here
    assert classify("Which terminal does Lufthansa use at Heathrow?", 2026) == "general_info"
    assert PENALTY == {"news": 2.0, "encyclopedic": 1.0, "general_info": 0.0}


def _pool(cands, r, snippet=True):
    return {"name": "p", "cands": cands, "r": r, "snippet": snippet}


def _cand(value, urls, ctx="x"):
    return {"value": value, "context": ctx, "url": urls[0], "urls": urls, "score": 1.0}


def test_consolidation_itemizes_confidence_counts_mirrors_once_and_flags_conflicts():
    from src.planck3.consolidate import consolidate, divergence
    from src.planck3.registry import trust_record
    tf = lambda d: trust_record(d)  # noqa: E731
    mirrors = _pool([_cand("1943", ["https://en.wikipedia.org/a", "https://www.wikiwand.com/a"])], [0.9])
    c = consolidate([mirrors], "year", "encyclopedic", tf)
    assert c["value"] == "1943" and len(c["families"]) == 1                           # Wikipedia + mirror = one source
    assert c["confidence"] == 8.0 and [round(v) for _, v in c["items"]] == [9, -1]   # 9, no agreement bonus, -1 type
    indep = _pool([_cand("1943", ["https://en.wikipedia.org/a", "https://www.britannica.com/a", "https://www.bbc.com/a"])], [0.9])
    c = consolidate([indep], "year", "encyclopedic", tf)
    assert c["confidence"] == 10.0 and len(c["families"]) == 3                        # 10 + 2 agree - 1, clamped
    conflict = _pool([_cand("1943", ["https://www.britannica.com/a"]), _cand("1948", ["https://www.bbc.com/b"])], [0.6, 0.35])
    c = consolidate([conflict], "year", "news", tf)
    assert c["value"] == "1943" and c["conflicts"][0]["value"] == "1948"
    assert c["confidence"] == 10 - 2 - 2 and c["band"] == "low"                       # conflict (BBC at 7) and news
    low_only = _pool([_cand("1950", ["https://rt.com/x"])], [0.9])
    assert consolidate([low_only], "year", "news", tf)["band"] == "sceptical"
    assert consolidate([_pool([_cand("x", ["https://a.org"])], [0.01])], "entity", "news", tf)["band"] == "none"
    # a contrarian user (personal = 10 - system) tilts the choice: it must be detected and shown
    split = _pool([_cand("Kyiv says A", ["https://kyivindependent.com/a"]), _cand("Moscow says B", ["https://rt.com/b"])], [0.5, 0.5])
    sysc = consolidate([split], "text", "news", tf)
    contra = consolidate([split], "text", "news", lambda d: trust_record(d, personal=10 - trust_record(d)["system"]))
    assert sysc["value"] == "Kyiv says A"
    d = divergence(contra, sysc, "text")
    assert contra["value"] == sysc["value"] or d["changed_value"]
    assert contra["confidence"] < sysc["confidence"] and d is not None and d["changed_band"]


def test_writer_faithfulness_and_template():
    from src.planck3.writer import faithful, source_name, template_text
    src = [{"domain": "bbc.com", "name": "BBC", "context": "IKEA was founded in 1943 by Ingvar Kamprad.", "says": None, "trust": 7},
           {"domain": "meduza.io", "name": "Meduza", "context": "Founded in 1943 in Sweden.", "says": None, "trust": 10}]
    assert faithful("1943, according to BBC [1] and Meduza [2].", "When was IKEA founded?", "1943", src, "year")[0]
    ok, probs = faithful("1943, according to BBC [1]; it opened 12 stores in Norway [3].", "When was IKEA founded?", "1943", src, "year")
    assert not ok and any("12" in p for p in probs) and any("Norway" in p for p in probs) and any("cites" in p for p in probs)
    assert not faithful("It was founded long ago [1].", "When was IKEA founded?", "1943", src, "year")[0]
    assert source_name("en.wikipedia.org") == "Wikipedia" and source_name("bbc.co.uk") == "BBC" and source_name("meduza.io") == "Meduza"
    for b, start in (("high", "1943."), ("good", "1943, according"), ("low", "Possibly"), ("sceptical", "Unclear")):
        t = template_text({"value": "1943", "band": b}, src)
        assert t.startswith(start) and faithful(t, "When was IKEA founded?", "1943", src, "year")[0]


def test_answer_pipeline_end_to_end_with_personal_tilt(tmp_path):
    from src.planck3.answer import Answerer
    from src.planck3.readers import HeuristicReader
    from src.planck3.writer import TemplateWriter
    st = Store(tmp_path / "s.sqlite")
    a = Answerer(SnippetWeb(), st, HeuristicReader(), TemplateWriter())
    r = a.run("What year was IKEA founded?")
    ex = r["explain"]
    assert r["value"] == "1943" and "1943" in r["answer_text"] and ex["pipeline"] == "answer"
    assert ex["qtype"] == "encyclopedic" and sum(v for _, v in ex["items"]) == pytest.approx(ex["confidence10"], abs=0.11)
    assert ex["steps"][0].startswith("Searched") and ex["writer_faithful"] and ex["divergence"] is None
    st.feedback("source", "example.org", 1)
    st.feedback("source", "example.org", 1)
    st.feedback("source", "example.org", 1)
    r2 = a.run("What year was IKEA founded?")
    t = next(s["trust"] for s in r2["explain"]["sources"] if s["domain"] == "example.org")
    assert t["personal"] == 10 and t["effective"] == 6.5 and t["conflict"]              # capped and flagged
    assert r2["explain"]["divergence"] is not None                                       # the band moved: shown on the card
    from src.planck3.chat import Turn, answer_card_text, render_why
    turn = Turn("q", "What year was IKEA founded?", "year", "IKEA", "NEW", r2)
    assert "Your source preferences changed this answer" in answer_card_text(turn)
    assert "you and the system disagree" in render_why(turn)
    st.close()


def test_sgs_writer_trains_and_generates_on_a_tiny_checkpoint(tmp_path):
    import json as _json
    import sentencepiece as spm
    import torch
    from src.planck3.writer import SGSWriter, train_writer
    from src.sgs_lm import SGSLanguageModel
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["QUESTION: When was IKEA founded? ANSWER: 1943 SOURCES: BBC says founded in 1943",
                                 "1943, according to BBC [1]. <END>", "the quick brown fox"] * 40), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / "tok"), vocab_size=80,
                                   model_type="unigram", hard_vocab_limit=False, user_defined_symbols=["<END>"])
    model = SGSLanguageModel(vocab_size=spm.SentencePieceProcessor(model_file=str(tmp_path / "tok.model")).get_piece_size(),
                             d_s=8, d_f=16, n_passes=2, n_heads=2, max_len=128)
    torch.save({"model": model.state_dict()}, tmp_path / "tiny.pt")
    data = tmp_path / "train.jsonl"
    row = {"prompt": "QUESTION: When was IKEA founded?\nANSWER: 1943\nCONFIDENCE: 8/10 good\nSOURCES:\n[1] BBC (7/10): founded in 1943\nANSWER TEXT:",
           "target": "1943, according to BBC [1].", "faithful": True}
    data.write_text("\n".join(_json.dumps(row) for _ in range(20)), encoding="utf-8")
    meta = train_writer(data, str(tmp_path / "tiny.pt"), str(tmp_path / "tok.model"), tmp_path / "w", "tiny", epochs=1, batch_size=4)
    assert meta["examples"] == 20 and (tmp_path / "w" / "best.pt").exists()
    w = SGSWriter(str(tmp_path / "w" / "best.pt"), device="cpu", max_new=8)
    out = w.write("When was IKEA founded?", "year", {"value": "1943", "confidence": 8.0, "band": "good",
                                                    "families": [], "conflicts": []}, {})
    assert out["text"] and (out["faithful"] or out["fallback"])                        # either faithful or the template


# ── round 6: KV cache, relevance-checked search, writer fixes, the cross-encoder reader ──
def test_kv_cache_is_exact_and_expands_for_a_shared_prefix():
    import torch
    from src.sgs_lm import SGSLanguageModel
    torch.manual_seed(0)
    m = SGSLanguageModel(vocab_size=97, d_s=8, d_f=24, n_passes=3, n_heads=2, max_len=48).eval()
    x = torch.randint(0, 97, (2, 20))
    with torch.no_grad():
        full = m(x)
        out, cache = m.forward_cached(x[:, :7])
        parts = [out]
        for t in range(7, 20):                                   # one token at a time, as in decoding
            o, cache = m.forward_cached(x[:, t:t + 1], cache)
            parts.append(o)
        assert torch.allclose(full, torch.cat(parts, 1), atol=1e-5) and cache["n"] == 20
        _, c1 = m.forward_cached(x[:1, :7], hidden=True)
        y = torch.randint(0, 97, (3, 5))
        h, _ = m.forward_cached(y, SGSLanguageModel.expand_cache(c1, 3), hidden=True)
        m.return_hidden = True
        ref = m(torch.cat([x[:1, :7].expand(3, 7), y], 1))[:, 7:]
        assert torch.allclose(h, ref, atol=1e-5)
    with pytest.raises(ValueError):
        m.forward_cached(torch.zeros(1, 49, dtype=torch.long))


def test_relevance_check_rejects_off_topic_results_and_walks_the_chain(tmp_path, monkeypatch):
    from src.planck3 import web as W
    q = "How many Riksdag seats did the left-wing opposition bloc win in Sweden's September 2026 general election?"
    junk = [{"url": f"https://junk{i}.com/x", "title": "Free videos", "snippet": "Calculator and deals", "domain": f"junk{i}.com"}
            for i in range(5)]
    good = [{"url": "https://dw.com/a", "title": "Sweden election: left-wing bloc wins Riksdag majority",
             "snippet": "The opposition bloc won 176 seats in the general election.", "domain": "dw.com"},
            {"url": "https://bbc.com/b", "title": "Swedish general election results", "snippet": "Left bloc 176 seats, Sweden.",
             "domain": "bbc.com"}]
    assert not W.on_topic(q, junk) and W.on_topic(q, good)
    assert "september" not in W.query_keys(q) and "2026" not in W.query_keys(q)
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    w = W.Web(W.WebCache(tmp_path / "c"), search_backend="ddgs")
    w.secondary = ["brave"]
    monkeypatch.setattr(w, "_search_ddgs", lambda qq: junk)
    monkeypatch.setattr(w, "_search_brave", lambda qq: good)
    monkeypatch.setattr(w, "_search_wikipedia", lambda qq: [])
    assert w.search(q) == good and w.last_backend == "brave" and w.last_on_topic
    h = w.search_health()
    assert h["primary_offtopic_rate"] == 1.0 and h["secondary_rate"] == 1.0 and h["on_topic_rate"] == 1.0
    w2 = W.Web(W.WebCache(tmp_path / "d"), search_backend="ddgs")
    monkeypatch.setattr(w2, "_search_ddgs", lambda qq: junk)
    monkeypatch.setattr(w2, "_search_wikipedia", lambda qq: [])
    assert w2.search(q) == junk and w2.last_on_topic is False                 # nothing on topic: served, but counted
    assert w2.search_health()["on_topic_rate"] == 0.0 and not w2.search_health()["valid"]
    w3 = W.Web(W.WebCache(tmp_path / "e"), search_backend="ddgs")
    w3.relevance = False                                                       # round 5 behaviour stays available
    monkeypatch.setattr(w3, "_search_ddgs", lambda qq: junk)
    assert w3.search(q) == junk and w3.search_health()["valid"]


def test_brave_quota_header_pauses_at_the_reserve(tmp_path, monkeypatch):
    from src.planck3 import web as W
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    w = W.Web(W.WebCache(tmp_path / "c"), search_backend="brave")
    w.brave_reserve = 100
    w.secondary = ["ddgs"]
    other = [{"url": "https://ikea.com/a", "title": "IKEA history", "snippet": "IKEA was founded in 1943.", "domain": "ikea.com"}]
    monkeypatch.setattr(w, "_search_ddgs", lambda q: other)

    class R:
        status_code = 200
        headers = {"X-RateLimit-Remaining": "1, 100"}

        def raise_for_status(self):
            pass

        def json(self):
            return {"web": {"results": [{"url": "https://www.ikea.com/b", "title": "IKEA", "description": "Founded 1943."}]}}
    monkeypatch.setattr(w.session, "get", lambda *a, **k: R())
    assert w.search("When was IKEA founded?")[0]["domain"] == "ikea.com" and w.brave_remaining == 100
    assert w._brave_paused                                                     # at the reserve: Brave is left alone
    assert w.search("Who founded IKEA?") == other and w.calls["search_quota_skips"] == 1


def test_writer_citation_groups_lead_and_tries(tmp_path):
    import json as _json
    from src.planck3.writer import TemplateWriter, cited, faithful, lead, lead_ok, rescore_data
    src = [{"domain": "bbc.com", "name": "BBC", "context": "IKEA was founded in 1943 by Ingvar Kamprad.", "says": None, "trust": 7},
           {"domain": "meduza.io", "name": "Meduza", "context": "Founded in 1943 in Sweden.", "says": None, "trust": 10}]
    assert cited("a [1, 2] b [1-2] c [2 and 1]") == [1, 2, 1, 2, 2, 1]
    assert faithful("1943, according to BBC and Meduza [1, 2].", "When was IKEA founded?", "1943", src, "year")[0]
    assert not faithful("1943 [1, 3].", "When was IKEA founded?", "1943", src, "year")[0]
    assert lead("low", "1943") == "Possibly 1943" and lead("high", "1943") == "1943" and lead("sceptical", "x") == "Unclear:"

    class Scripted(TemplateWriter):
        name, tries = "scripted", 3

        def __init__(self, outs):
            super().__init__()
            self.outs, self.seen = outs, []

        def candidates(self, prompt, lead_text):
            for o in self.outs:
                self.seen.append(lead_text)
                yield o
    cons = {"value": "1943", "confidence": 8.0, "band": "good", "families": [
        {"domain": "bbc.com", "url": "https://bbc.com/a", "context": src[0]["context"], "trust": {"effective": 7}}], "conflicts": []}
    w = Scripted(["1943, says Norway [1].", "1943, according to BBC [1].", "never reached"])
    out = w.write("When was IKEA founded?", "year", cons, {})
    assert out["text"] == "1943, according to BBC [1]." and out["tries"] == 2 and not out["first_try_faithful"]
    assert out["faithful"] and not out["fallback"] and w.seen == ["1943", "1943"]       # stops at the first faithful
    bad = Scripted(["It is old [1]."]).write("When was IKEA founded?", "year", cons, {})
    assert bad["fallback"] and bad["text"] == bad["template"]
    data = tmp_path / "w.jsonl"
    prompt = "QUESTION: When was IKEA founded?\nANSWER: 1943\nCONFIDENCE: 8/10 good\nSOURCES:\n[1] BBC (7/10): IKEA was founded in 1943.\n[2] Meduza (10/10): Founded 1943.\nANSWER TEXT:"
    data.write_text(_json.dumps({"prompt": prompt, "band": "good", "target": "1943, according to BBC and Meduza [1, 2].",
                                 "faithful": False}) + "\n", encoding="utf-8")
    assert rescore_data(data) == {"examples": 1, "faithful": 1, "faithful_and_lead": 1}       # round 5's checker bug undone
    row = _json.loads(data.read_text(encoding="utf-8"))
    assert row["faithful"] and row["faithful_r5"] is False and lead_ok(row)


def test_sgs_writer_decodes_with_the_cache_identically(tmp_path):
    import json as _json
    import sentencepiece as spm
    import torch
    from src.planck3.writer import SGSWriter, kv_check
    from src.sgs_lm import SGSLanguageModel
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["QUESTION: When was IKEA founded? ANSWER: 1943 SOURCES: BBC says founded in 1943",
                                 "1943, according to BBC [1]. <END>", "Possibly 1943 the quick brown fox"] * 40), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / "tok"), vocab_size=80,
                                   model_type="unigram", hard_vocab_limit=False, user_defined_symbols=["<END>"])
    torch.manual_seed(0)
    model = SGSLanguageModel(vocab_size=spm.SentencePieceProcessor(model_file=str(tmp_path / "tok.model")).get_piece_size(),
                             d_s=8, d_f=16, n_passes=2, n_heads=2, max_len=128)
    torch.save({"model": model.state_dict(), "tokenizer": str(tmp_path / "tok.model")}, tmp_path / "w.pt")
    prompts = [("QUESTION: When was IKEA founded?\nANSWER: 1943\nANSWER TEXT:", "1943"),
               ("QUESTION: When was IKEA founded?\nANSWER: 1943\nANSWER TEXT:", "Possibly 1943")]
    res = kv_check(str(tmp_path / "w.pt"), prompts, device="cpu")
    assert res["identical"] == 1.0 and res["prompts"] == 2
    w = SGSWriter(str(tmp_path / "w.pt"), device="cpu", max_new=6)
    outs = list(w.candidates(prompts[1][0], prompts[1][1]))
    assert len(outs) == 3 and all(o.startswith("Possibly 1943") for o in outs)        # every try starts from the lead
    assert outs == list(w.candidates(prompts[1][0], prompts[1][1]))                     # seeded: reproducible


def test_portal_parsing_and_news_question_checks():
    from src.planck3.readtasks import check_news_question, named_in_bench, parse_portal, strip_wiki
    wt = ("'''Disasters and accidents'''\n*[[2025 floods]]\n**At least 31 people are killed when a [[bus]] falls into a ravine "
          "in southern [[Bolivia]]. [https://reuters.com/x (Reuters)]\n'''Sports'''\n*[[Tadej Pogačar]] wins the [[2025 Tour de France]], "
          "his fourth title in the race.<ref>x</ref>\n'''Unknown section'''\n*Ignored item that is long enough to be kept otherwise ok.")
    items = parse_portal(wt)
    assert [i["section"] for i in items] == ["disaster", "sport"] and "Reuters" not in items[0]["text"]
    assert strip_wiki("[[Kalat, Balochistan|Kalat]] ''x''") == "Kalat x"
    it = {"id": "2025-07-27_sport", "date": "2025-07-27", "section": "sport", "text": items[1]["text"]}
    ok, why = check_news_question({"question": "Who won the Tour de France in July 2025?", "answer": "Tadej Pogačar",
                                   "answer_type": "entity"}, it)
    assert ok and ok["gold"] == ["Tadej Pogačar"] and ok["id"] == "news_2025-07-27_sport" and why == "ok"
    for obj, reason in (({"question": "Who won the Tour de France?", "answer": "Tadej Pogačar", "answer_type": "entity"}, "no year"),
                        ({"question": "Did Tadej Pogačar win the 2025 Tour de France race?", "answer": "Tadej Pogačar",
                          "answer_type": "entity"}, "answer in the question"),
                        ({"question": "Who won the 2025 Tour de France race in July?", "answer": "Jonas Vingegaard",
                          "answer_type": "entity"}, "not in the item")):
        task, why = check_news_question(obj, it)
        assert task is None and reason in why
    assert named_in_bench("Dubai International Airport", ["which terminal does emirates use at dubai international airport dxb"])
    assert not named_in_bench("Heathrow Airport", ["which terminal does emirates use at dubai international airport dxb"])


def _tiny_lm(tmp_path, texts):
    import sentencepiece as spm
    import torch
    from src.sgs_lm import SGSLanguageModel
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(texts * 20), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / "tok"), vocab_size=120,
                                   model_type="unigram", hard_vocab_limit=False)
    torch.manual_seed(0)
    m = SGSLanguageModel(vocab_size=spm.SentencePieceProcessor(model_file=str(tmp_path / "tok.model")).get_piece_size(),
                         d_s=8, d_f=16, n_passes=2, n_heads=2, max_len=256)
    torch.save({"model": m.state_dict()}, tmp_path / "base.pt")
    return str(tmp_path / "base.pt"), str(tmp_path / "tok.model")


def test_cross_reader_trains_once_calibrates_and_reads_with_the_prefix_cache(tmp_path):
    import json as _json
    from src.planck3 import xreader as X
    pts = _synthetic_points(n=120)
    for i, p in enumerate(pts):
        p["task_id"] = ("news_" if i % 3 == 0 else "recent_" if i % 3 == 1 else "t") + str(i)
    path = tmp_path / "points.jsonl"
    path.write_text("\n".join(_json.dumps(p) for p in pts), encoding="utf-8")
    base, tok = _tiny_lm(tmp_path, [X.prefix_text(p["question"], "year") + X.cand_text(c, True) for p in pts[:30] for c in p["cands"]])
    meta = X.train([path], base, tok, tmp_path / "xr", "tiny", epochs=1, cands_per_step=16)
    assert meta["n_points"]["test"] > 0 and 0 <= meta["test"]["choice_acc"] <= 1 and meta["temperature"] > 0
    assert set(meta["test_by_group"]) <= set(X.GROUPS) and (tmp_path / "xr" / "reader.pt").exists()
    pool = {"cands": pts[0]["cands"] * 5, "snippet": True}                     # 20 candidates: top 16 read, rest 0
    rd_cache = X.CrossReader(str(tmp_path / "xr" / "reader.pt"), device="cpu", prefix_cache=True)
    rd_full = X.CrossReader(str(tmp_path / "xr" / "reader.pt"), device="cpu", prefix_cache=False)
    a = rd_cache.logits(pts[0]["question"], "year", pool)
    b = rd_full.logits(pts[0]["question"], "year", pool)
    assert np.allclose(a, b, atol=1e-4)                                         # prefix cache = full forward
    r = rd_cache.read(pts[0]["question"], "year", pool)
    assert len(r) == 20 and r[16:] == [0.0] * 4 and 0 < sum(r) <= 1 + 1e-6
    assert X.shift_years("In 2025 (not 1999) and 20255", 1) == "In 2026 (not 2000) and 20255"
    assert X.split_of("news_x") == X.split_of("news_x") and X.group_of("info_iata001") == "info" and X.group_of("twinner1") == "stable"


def test_round6_product_rules(tmp_path):
    from src.planck3 import round6 as R6
    from src.planck3.util import write_json
    def log(b, acc):
        write_json(tmp_path / f"xreader_{b}" / "train_log.json", {"test": {"choice_acc": acc}})
    log("planck", 0.70)
    log("hertz", 0.72)
    assert R6.choose_reader(tmp_path)["reader"] == "x-planck"                  # +2 points does not justify 6x the size
    log("hertz", 0.74)
    assert R6.choose_reader(tmp_path)["reader"] == "x-hertz"
    write_json(tmp_path / "r6_x-hertz_planck" / "summary.json", {"writer_faithful": 0.93})
    write_json(tmp_path / "r6_x-hertz_hertz" / "summary.json", {"writer_faithful": 0.96})
    d = R6.product_decision(tmp_path)
    assert d["writer"] == "hertz" and d["run"] == "r6_x-hertz_hertz"            # smallest writer that passes W
    write_json(tmp_path / "r6_x-hertz_hertz" / "summary.json", {"writer_faithful": 0.90})
    assert R6.product_decision(tmp_path)["writer"] == "template"


def test_round6b_pools_as_served_and_rule_r(tmp_path):
    import json
    from src.planck3 import round6 as R6
    from src.planck3 import xreader as X
    # a right candidate beyond the top 16 is NOT moved into the last slot (round 6's last-slot leak)
    p = {"cands": [{"value": str(i)} for i in range(20)], "labels": [False] * 18 + [True, False]}
    cands, labels = X.pool_cands(p)
    assert [c["value"] for c in cands] == [str(i) for i in range(16)] and not any(labels)
    assert X.right_beyond_cut(p) and not X.right_beyond_cut(dict(p, labels=[True] + [False] * 19))

    def run(name, right):
        (tmp_path / name).mkdir()
        with open(tmp_path / name / "results.jsonl", "w") as f:
            for i, ok in enumerate(right):
                f.write(json.dumps({"task_id": f"t{i}", "shown_correct": ok}) + "\n")
    n = 100
    run("r6_planck_template", [i < 50 for i in range(n)])
    run("r6_gemma_template", [i < 70 for i in range(n)])
    run("r6b_x-planck_template", [i < 62 for i in range(n)])               # 0.62 >= midpoint 0.60, +12 only-right
    r = R6.rule_r(tmp_path, "r6b_x-planck_template")
    assert r["pass"] and abs(r["midpoint"] - 0.60) < 1e-9 and r["p"] < 0.05
    run("r6b_x-hertz_template", [i < 55 for i in range(n)])                 # better, below the midpoint
    assert not R6.rule_r(tmp_path, "r6b_x-hertz_template")["pass"]
    assert R6.rule_r(tmp_path, "missing")["pass"] is False


def test_round7_reading_sets_become_reader_pools(tmp_path, monkeypatch):
    from src.planck3 import rcdata as RC
    assert RC.gold_type("1858") == "year" and RC.gold_type("740,000") == "number" and RC.gold_type("Botswana") == "entity"
    assert RC.gold_type("soprano") is None and RC.answer_type_of("As a vocal range in opera", ["soprano"]) is None
    docs = RC.parse_docs("[DOC] [TLE] Vocal range - Wikipedia [PAR] Soprano is the highest range. [DOC] [TLE] Opera [PAR] A <b>tenor</b> sings.")
    assert [d["title"] for d in docs] == ["Vocal range - Wikipedia", "Opera"] and docs[1]["text"] == "A tenor sings."
    page = {"qid": "q1", "question": "Who founded the Acme Rocket Company?",
            "context": "The Acme Rocket Company was founded by Jane Holloway in 1931. Its rival was set up by Tom Baker.",
            "answers": ["Jane Holloway"]}
    p = RC.pool_of(page, "squad")
    assert p["answer_type"] == "entity" and p["source"] == "page" and p["task_id"].startswith("squad_")
    assert any(p["labels"]) and p["cands"][p["labels"].index(True)]["value"] == "Jane Holloway"
    snip = {"qid": "q2", "question": "Who founded the Acme Rocket Company?", "answers": ["Jane Holloway"],
            "context": "[DOC] [TLE] Acme Rocket Company [PAR] Founded by Jane Holloway. [DOC] [TLE] Acme history [PAR] Jane Holloway set it up."}
    q = RC.pool_of(snip, "searchqa")
    assert q["source"] == "snippet" and any(q["labels"])
    neg = RC.pool_of(dict(page, answers=[]), RC.NEG_GROUP)
    assert neg is not None and not any(neg["labels"])                           # SQuAD 2.0: the answer is not in here
    assert RC.pool_of(dict(page, answers=[]), "squad") is None
    # benchmark subjects are dropped: "Heathrow Express" is a general-information benchmark question
    bench = RC.bench_index()
    assert RC.in_bench("Where does the Heathrow Express terminate?", ["Paddington"], bench)
    assert not RC.in_bench("Who founded the Acme Rocket Company?", ["Jane Holloway"], bench)
    rows = {g: [] for g in RC.GROUPS}
    rows["squad"] = [dict(page, qid=f"s{i}") for i in range(10)] + [dict(page, qid=f"n{i}", answers=["Nobody Here"]) for i in range(10)]
    rows[RC.NEG_GROUP] = [dict(page, qid=f"x{i}", answers=[]) for i in range(5)]
    monkeypatch.setattr(RC, "_rows_by_set", lambda seed, take: rows)
    h = RC.build(tmp_path / "rc.jsonl", per_set=10, neg=3)
    s = h["sets"]["squad"]
    assert s["with_right_in_top16"] == 6 and s["without"] == 4                  # at most 40% without a right one
    assert h["sets"][RC.NEG_GROUP]["kept"] == 3 and (tmp_path / "rc.health.json").exists()


def test_round7_two_stage_reader_and_rules(tmp_path):
    import json as _json
    from src.planck3 import round7 as R7
    from src.planck3 import xreader as X
    from src.planck3.util import write_json
    pts = _synthetic_points(n=90)
    for i, p in enumerate(pts):
        p["task_id"] = f"newsqa_{i}"
    path = tmp_path / "rc.jsonl"
    path.write_text("\n".join(_json.dumps(p) for p in pts), encoding="utf-8")
    base, tok = _tiny_lm(tmp_path, [X.prefix_text(p["question"], "year") + X.cand_text(c, True) for p in pts[:30] for c in p["cands"]])
    a = X.train([tmp_path / "none", path], base, tok, tmp_path / "xa", "tiny", epochs=1, cands_per_step=16, max_eval=5)
    assert a["n_points"]["val"] <= 5 and a["n_points"]["test"] <= 5 and "vs_ranker_p" in a["test"]
    b = X.train([path], base, tok, tmp_path / "xb", "tiny", epochs=1, cands_per_step=16, init=str(tmp_path / "xa" / "reader.pt"))
    assert b["init"].endswith("reader.pt")
    # rules: N needs +10 points over the ranker on NewsQA with p < 0.05
    write_json(tmp_path / "xreader_planck_rc" / "train_log.json", {"test_by_group": {"newsqa": {
        "choice_acc": 0.62, "ranker_acc": 0.45, "vs_ranker_p": 0.001, "only_choice_right": 60, "only_ranker_right": 20,
        "pools_with_a_right_candidate": 400}}})
    assert R7.rule_n(tmp_path)["pass"]
    write_json(tmp_path / "xreader_planck_rc" / "train_log.json", {"test_by_group": {"newsqa": {
        "choice_acc": 0.50, "ranker_acc": 0.45, "vs_ranker_p": 0.04, "only_choice_right": 40, "only_ranker_right": 20,
        "pools_with_a_right_candidate": 400}}})
    assert not R7.rule_n(tmp_path)["pass"]                                      # +5 points is not "reads news"
    c1 = R7.rule_c1({"bands_monotone": True, "bands": {"high": {"n": 25, "precision": 0.92}, "good": {"n": 10, "precision": 0.5}}})
    assert c1["pass"] and not c1["good"]["judged"]                               # under 20 answers: not judged
    d = R7.decide(tmp_path)
    assert d["reader"] == "x-planck" and d["outcome"].startswith("rebuild") and (tmp_path / "r7_product.json").exists()


def test_round8_schedule_mobile_reader_and_rules(tmp_path):
    import json as _json
    from src.planck3 import round8 as R8
    from src.planck3 import xreader as X
    from src.planck3.util import write_json
    # the cosine schedule is sized in batches, exactly as _batches packs them (round 7 undercounted)
    sizes = [16, 16, 16, 16, 5, 16, 3, 16, 16, 16, 16, 16]
    fake = [type("E", (), {"conts": [0] * k})() for k in sizes]
    assert X._n_batches(sizes, 64) == len(list(X._batches(fake, 64))) == 3
    assert X._n_batches([], 64) == 0
    # the reader on the phone proxy: int8 weights, top-k candidates, its size on the device
    pts = _synthetic_points(n=60)
    path = tmp_path / "rc.jsonl"
    path.write_text("\n".join(_json.dumps(p) for p in pts), encoding="utf-8")
    base, tok = _tiny_lm(tmp_path, [X.prefix_text(p["question"], "year") + X.cand_text(c, True) for p in pts[:20] for c in p["cands"]])
    X.train([path], base, tok, tmp_path / "xa", "tiny", epochs=1, cands_per_step=16)
    rd = X.CrossReader(str(tmp_path / "xa" / "reader.pt"), device="cpu", quant="int8", max_cands=2)
    p0 = pts[0]
    r = rd.read(p0["question"], "year", {"cands": p0["cands"], "snippet": True})
    assert len(r) == len(p0["cands"]) and r[2:] == [0.0, 0.0] and rd.config["quant"] == "int8" and rd.config["size_mb"] > 0
    # rule M: an int8 configuration within 5 s, 1 GB and 2 points of the GPU run on >= 200 questions
    def run(name, right, extra=None):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.jsonl").write_text("\n".join(_json.dumps({"task_id": f"t{i}", "shown_correct": i < right}) for i in range(243)), encoding="utf-8")
        write_json(d / "summary.json", {"accuracy": right / 243, **(extra or {})})
    run("r7_x-hertz-rc_template", 146)
    run("r8_x-hertz-rc_cpu-int8-top16", 145, {"timing_ms_per_task": {"read_ms": 3100}, "reader_config": {"quant": "int8", "size_mb": 690, "max_cands": 16}})
    run("r8_x-hertz-rc_cpu-int8-top8", 120, {"timing_ms_per_task": {"read_ms": 1500}, "reader_config": {"quant": "int8", "size_mb": 690, "max_cands": 8}})
    m = R8.rule_m(tmp_path)
    assert m["pass"] and m["passing"] == ["cpu-int8-top16"] and not m["configs"]["cpu-int8-top8"]["pass"]   # top 8 loses 10 points
    assert m["configs"]["cpu-fp32-top16"]["why"] == "not run"
    d = R8.decide(tmp_path, ckpts=tmp_path / "ck")
    assert d["outcome"].startswith("pending") and d["rule_m"]["pass"]           # mid-run: only the mobile check exists
    run(R8.RUN0, 130)
    d = R8.decide(tmp_path, ckpts=tmp_path / "ck")
    assert d["outcome"].startswith("size") and "M passes" in d["outcome"] and (tmp_path / "r8_product.json").exists()
    assert d["reader_checkpoint"].endswith("xreader_planck_p15rc/reader.pt") and "why" in d["base"]


def test_round9_brave_check_needs_the_live_api(tmp_path, monkeypatch):
    from src.planck3 import web as W
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    w = W.Web(W.WebCache(tmp_path / "c"), search_backend="brave")
    w.cache.put("search", "brave|Wikipedia", {"results": [{"url": "https://w.org", "title": "W", "snippet": "W", "domain": "w.org"}]})

    class R:
        def __init__(self, code, rem):
            self.status_code, self.headers = code, {"X-RateLimit-Remaining": f"1, {rem}"}

        def raise_for_status(self):
            pass

        def json(self):
            return {"web": {"results": [{"url": "https://w.org/x", "title": "W", "description": "W"}]}}
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    monkeypatch.setattr(w.session, "get", lambda *a, **k: R(429, 0))
    assert not w.brave_alive()                          # a cached answer no longer passes at 0 quota (round 6's bug)
    assert w.brave_quota()["why"] == "quota used up this month"
    w2 = W.Web(W.WebCache(tmp_path / "c"), search_backend="brave")
    monkeypatch.setattr(w2.session, "get", lambda *a, **k: R(200, 1500))
    q = w2.brave_quota()
    assert q["ok"] and q["remaining"] == 1500 and w2.brave_alive()
    w3 = W.Web(W.WebCache(tmp_path / "c", mode="replay"), search_backend="brave")
    assert w3.brave_alive()                             # replay: the cache is all there is


def test_round9_fresh_benchmark_and_rules(tmp_path, monkeypatch):
    import json as _json
    from datetime import date
    from src.planck3 import benchgen as bg
    from src.planck3 import freshbench as FB
    from src.planck3 import readtasks as RT
    from src.planck3 import round9 as R9
    from src.planck3.util import write_json
    # Wikidata: 2026 items unused before; an item already in a benchmark is dropped by QID
    used = sorted(FB.used_qids_and_texts()[0])[0]
    rows = [{"item": f"http://www.wikidata.org/entity/{q}", "itemLabel": lab, "ans": "x", "ansLabel": "Team A", "aliases": ""}
            for q, lab in ((used, "2026 Old Cup"), ("Q999000001", "2026 Zebra Cup"), ("Q999000002", "Quokkaville Zither Invitational"))]
    monkeypatch.setattr(bg, "sparql", lambda q: rows if "P1346" in q and "P585" in q else [])
    monkeypatch.setattr(FB, "WD_PER", {"winner": 5, "election": 1, "season": 1, "headgov": 1})
    tasks, st = FB.wikidata_fresh()
    assert {t["qid"] for t in tasks} == {"Q999000001", "Q999000002"} and st["winner"]["used before"] == 1
    assert all(t["regime"] == "fresh" and t["id"].startswith("f9winner") for t in tasks)
    # the portal: several items a day, round-robin across sections
    text = "\n".join(["'''Sports'''", "* Team Alpha wins the 2026 Northern Cup final in Oslo, Norway, beating Team Beta 2-1.",
                      "* Runner Carla Dee sets a 2026 world record of 2:01:03 in the Berlin Marathon in Germany.",
                      "'''Law and crime'''", "* A court in Lima, Peru, sentences former mayor Juan Ruiz to 12 years in prison in 2026."])
    monkeypatch.setattr(RT, "_wikitext", lambda title: text)
    items = FB.portal_items_after(date(2026, 10, 8), date(2026, 10, 9), per_day=3)
    assert len(items) == 3 and {i["section"] for i in items} == {"sport", "law"} and items[0]["date"] == "2026-10-09"
    # the file is the benchmark of record: never rebuilt once it exists
    out = tmp_path / "f9.json"
    write_json(out, {"created": "2026-10-19", "tasks": [{"id": "a"}]})
    assert FB.build("none", out)["tasks"] == [{"id": "a"}]
    # recalibrated bands: thresholds fitted on old answers
    old = [{"confidence": c, "shown_correct": ok} for c, ok in [(9, True)] * 19 + [(9, False)] + [(7, True)] * 15 + [(7, False)] * 5 + [(5, False)] * 10]
    th = R9.fit_thresholds(old)
    assert th["high"] == 9 and th["good"] == 7
    # R9 on fresh questions: the product vs round 5's reader and Gemma; Q vs its own GPU run
    def run(name, right, n=150):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.jsonl").write_text("\n".join(_json.dumps({"task_id": f"t{i}", "regime": "fresh" if i % 2 else "news",
                                                                "shown_correct": i < right, "confidence": 9 if i < right else 6})
                                                   for i in range(n)), encoding="utf-8")
        write_json(d / "summary.json", {"accuracy": right / n, "timing_ms_per_task": {"read_ms": 1400}, "bands": {}, "bands_monotone": True})
    run(R9.PRODUCT, 95)
    run(R9.BASE, 70)
    run(R9.REF, 100)
    run(R9.GPU, 96)
    run(R9.FLOOR, 60)
    run(R9.OLD_PRODUCT, 140, 243)
    d = R9.decide(tmp_path)
    assert d["rule_r"]["pass"] and d["rule_q"]["pass"] and d["rule_l"]["pass"] and d["outcome"].startswith("ship")
    assert d["reader_quant"] == "int8" and d["by_regime"][R9.PRODUCT]["fresh"]["n"] == 75
    run(R9.PRODUCT, 80)
    assert R9.decide(tmp_path)["outcome"].startswith("no ship")      # 80 < midpoint 85


def test_round9_brave_credit_plan_reports_no_monthly_quota(tmp_path, monkeypatch):
    from src.planck3 import web as W
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    monkeypatch.setattr(W.time, "sleep", lambda s: None)

    class R:
        def __init__(self, code, lim, rem):
            self.status_code, self.headers = code, {"X-RateLimit-Limit": lim, "X-RateLimit-Remaining": rem}

        def raise_for_status(self):
            pass

        def json(self):
            return {"web": {"results": [{"url": "https://w.org/x", "title": "W", "description": "W"}]}}
    # a credit-based plan: monthly limit 0, monthly remaining 0, and the search succeeds
    w = W.Web(W.WebCache(tmp_path / "c"), search_backend="brave")
    w.brave_reserve = 400
    monkeypatch.setattr(w.session, "get", lambda *a, **k: R(200, "1, 0", "1, 0"))
    q = w.brave_quota()
    assert q["ok"] and q["remaining"] is None and q["headers"]["X-RateLimit-Limit"] == "1, 0"
    assert not w._brave_paused                          # round 6 paused here and collected on ddgs instead
    # a plan with a real monthly quota that is used up: still read as used up
    w2 = W.Web(W.WebCache(tmp_path / "c2"), search_backend="brave")
    monkeypatch.setattr(w2.session, "get", lambda *a, **k: R(429, "1, 2000", "1, 0"))
    q2 = w2.brave_quota()
    assert not q2["ok"] and q2["remaining"] == 0 and q2["why"] == "quota used up this month"


def test_round10_brave_budget_soft_labels_and_rules(tmp_path, monkeypatch):
    import json as _json
    from src.planck3 import g2 as G
    from src.planck3 import round10 as R10
    from src.planck3 import web as W
    from src.planck3 import xreader as X
    from src.planck3.util import write_json
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    # a runner-wide ledger caps live Brave calls; collection stops (never falls back) and resumes
    ledger = tmp_path / "ledger.json"
    ledger.write_text(_json.dumps({"budget": 4, "used": 0}), encoding="utf-8-sig")
    monkeypatch.setenv("PLANCK3_BRAVE_LEDGER", str(ledger))
    w = W.Web(W.WebCache(tmp_path / "c"), search_backend="brave")
    w.fallback, w.relevance = False, False
    w.attach_ledger()
    res = [{"url": "https://a.org/x", "title": "Zed Cup", "snippet": "Team Alpha won the 2025 Zed Cup.", "domain": "a.org"}]
    monkeypatch.setattr(w, "_search_brave", lambda q: (setattr(w, "brave_used", w.brave_used + 1), w._ledger_write(), res)[2])
    monkeypatch.setattr(w, "fetch", lambda url: {})
    tasks = [{"id": f"recent_t{i}", "family": "fact", "question": f"Who won the 2025 Zed Cup {i}?", "answer_type": "entity",
              "gold": ["Team Alpha"], "regime": "recent"} for i in range(6)]
    out = tmp_path / "pb.jsonl"
    G.collect(tasks, w, out)
    prog = _json.loads(out.with_suffix(".progress.json").read_text(encoding="utf-8"))
    assert prog["stopped_for_credit"] and not prog["complete"] and prog["through"] == 3      # stops with 1 call left
    assert _json.loads(ledger.read_text(encoding="utf-8"))["used"] == 3
    ledger.write_text(_json.dumps({"budget": 100, "used": 0}), encoding="utf-8")
    w2 = W.Web(W.WebCache(tmp_path / "c"), search_backend="brave")
    w2.fallback, w2.relevance = False, False
    w2.attach_ledger()
    monkeypatch.setattr(w2, "_search_brave", lambda q: res)
    monkeypatch.setattr(w2, "fetch", lambda url: {})
    G.collect(tasks, w2, out)
    prog = _json.loads(out.with_suffix(".progress.json").read_text(encoding="utf-8"))
    assert prog["complete"] and prog["through"] == 6 and len({p["task_id"] for p in map(_json.loads, out.read_text().splitlines())}) == 6
    # Gemma's soft labels: stored on each pool, then used by the loss
    class FakeGemma:
        name = "gemma"
        usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

        def _complete(self, system, user):
            return '{"k": 0, "p": 0.8}'
    info = X.teach(out, FakeGemma())
    pts = [_json.loads(x) for x in out.read_text().splitlines()]
    assert info["taught"] == len(pts) and all(p["teacher"]["k"] == 0 and p["teacher"]["p"] == 0.8 for p in pts)
    assert X.teach(out, FakeGemma())["taught"] == len(pts)                   # resumable: nothing re-read
    import torch
    e = type("E", (), {})()
    e.conts, e.labels, e.teacher = [[1], [2], [3]], [False, True, False], (0, 0.8)
    lg = torch.zeros(3, 2, requires_grad=True)
    assert X._losses(lg, [e], 0.5).item() > X._losses(lg, [e], 0.0).item()   # the soft term is added
    # rule G: >= 5 points over the shipped reader on the new fresh questions, p < 0.05
    def run(name, right, n=200):
        d = tmp_path / "res" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.jsonl").write_text("\n".join(_json.dumps({"task_id": f"t{i}", "regime": "fresh", "shown_correct": i < right,
                                                                "confidence": 7, "changed_by_personal": False, "parts": []})
                                                   for i in range(n)), encoding="utf-8")
        write_json(d / "summary.json", {"accuracy": right / n, "timing_ms_per_task": {"read_ms": 3000}, "bands": {}, "bands_monotone": True})
    R = tmp_path / "res"
    run(R10.SHIPPED, 100)
    run(R10.PRODUCT, 125)
    run(R10.GPU, 126)
    run(R10.REF, 125)
    run(R10.PRODUCT + "_contrarian", 125)
    d = R10.decide(R)
    assert d["rule_g"]["pass"] and d["gap_closed"]["share"] == 1.0 and d["rule_r"]["pass"] and d["outcome"].startswith("ship")
    run(R10.PRODUCT, 108)
    assert not R10.decide(R)["rule_g"]["pass"]                               # +4 points is not enough
