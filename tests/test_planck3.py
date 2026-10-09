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
