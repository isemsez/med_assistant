import json
import tempfile
import unittest
from pathlib import Path

import httpx

from medassistant import gindex, health

EP = {"base_url": "http://x", "api_key": "k", "model": "m", "extra_body": {}}


class FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


class TestCheckModel(unittest.TestCase):
    def setUp(self):
        self._get = httpx.get

    def tearDown(self):
        httpx.get = self._get
        health._cache.clear()

    def test_ok_openai_shape(self):
        httpx.get = lambda *a, **kw: FakeResp(
            payload={"data": [{"id": "m"}, {"id": "other"}]})
        self.assertTrue(health.check_model(EP)["ok"])

    def test_ok_alias_without_tag(self):
        httpx.get = lambda *a, **kw: FakeResp(
            payload={"data": [{"id": "m:latest"}]})
        self.assertTrue(health.check_model(EP)["ok"])

    def test_model_missing(self):
        httpx.get = lambda *a, **kw: FakeResp(payload={"data": [{"id": "x"}]})
        r = health.check_model(EP)
        self.assertFalse(r["ok"])
        self.assertIn("error", r)

    def test_unconfigured(self):
        r = health.check_model({"base_url": "", "model": "", "api_key": ""})
        self.assertFalse(r["ok"])

    def test_connection_error(self):
        def boom(*a, **kw):
            raise httpx.ConnectError("refused")
        httpx.get = boom
        self.assertFalse(health.check_model(EP, timeout=0.5)["ok"])


class TestCheckEmbed(unittest.TestCase):
    def setUp(self):
        self._post = httpx.post

    def tearDown(self):
        httpx.post = self._post
        health._cache.clear()

    def test_ok(self):
        httpx.post = lambda *a, **kw: FakeResp(
            payload={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})
        r = health.check_embed(EP)
        self.assertTrue(r["ok"])
        self.assertEqual(r["dim"], 2)

    def test_empty_embedding(self):
        httpx.post = lambda *a, **kw: FakeResp(payload={"data": [{"embedding": []}]})
        self.assertFalse(health.check_embed(EP)["ok"])


class TestCheckIndex(unittest.TestCase):
    DOCS = {"1_1": {"code_version": "1_1", "name": "Грипп", "url": "u"}}

    def build(self, tmp, docs, stamp=True):
        d = Path(tmp)
        d.mkdir(parents=True, exist_ok=True)
        (d / "docs_meta.json").write_text(
            json.dumps(list(docs.values()), ensure_ascii=False), encoding="utf-8")
        idx = d / "index"
        idx.mkdir(exist_ok=True)
        meta = {"dim": 4, "rows": [{"code_version": "1_1", "name": "Грипп",
                                    "text": "Грипп"}]}
        if stamp:
            meta["docs_meta_sha1"] = gindex.docs_hash(docs)
            meta["docs_meta_count"] = len(docs)
        (idx / "meta.json").write_text(json.dumps(meta, ensure_ascii=False),
                                       encoding="utf-8")
        return d / "docs_meta.json", idx

    def test_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            dm, idx = self.build(tmp, self.DOCS)
            r = health.check_index(index_dir=idx, docs_meta_path=dm)
            self.assertTrue(r["ok"])
            self.assertTrue(r["fresh"])
            self.assertEqual(r["docs_meta"], 1)
            self.assertEqual(r["vectors"], 1)

    def test_stale_after_docs_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            dm, idx = self.build(tmp, self.DOCS)
            docs2 = dict(self.DOCS, **{"2_1": {"code_version": "2_1",
                                               "name": "ХСН", "url": "u2"}})
            Path(dm).write_text(json.dumps(list(docs2.values()), ensure_ascii=False),
                                encoding="utf-8")
            r = health.check_index(index_dir=idx, docs_meta_path=dm)
            self.assertFalse(r["ok"])
            self.assertFalse(r["fresh"])
            self.assertIn("build-index", r["error"])

    def test_legacy_index_without_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            dm, idx = self.build(tmp, self.DOCS, stamp=False)
            r = health.check_index(index_dir=idx, docs_meta_path=dm)
            self.assertTrue(r["ok"])
            self.assertIsNone(r["fresh"])
            self.assertIn("note", r)

    def test_index_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            dm, idx = self.build(tmp, self.DOCS)
            (idx / "meta.json").unlink()
            r = health.check_index(index_dir=idx, docs_meta_path=dm)
            self.assertFalse(r["ok"])

    def test_docs_meta_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            dm, idx = self.build(tmp, self.DOCS)
            Path(dm).unlink()
            r = health.check_index(index_dir=idx, docs_meta_path=dm)
            self.assertFalse(r["ok"])


class TestCheckCorpus(unittest.TestCase):
    def test_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.pdf").write_bytes(b"x")
            r = health.check_corpus(clinrec_dir=tmp)
            self.assertTrue(r["ok"])
            self.assertEqual(r["pdfs"], 1)

    def test_empty_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(health.check_corpus(clinrec_dir=tmp)["ok"])
        self.assertFalse(health.check_corpus(clinrec_dir=tmp + "/nope")["ok"])


class TestRunHealth(unittest.TestCase):
    def test_aggregates_and_caches(self):
        calls = {"n": 0}

        def counting(fn):
            def wrapper(*args, **kw):
                calls["n"] += 1
                return fn(*args, **kw)
            return wrapper

        orig = health.check_model, health.check_embed
        health.check_model = counting(lambda *a, **k: {"ok": True})
        health.check_embed = counting(lambda *a, **k: {"ok": True})
        try:
            r1 = health.run_health(cache_ttl=30)
            r2 = health.run_health(cache_ttl=30)  # served from cache
            self.assertTrue(r1["ok"] and r2["ok"])
            self.assertEqual(calls["n"], 3)  # model x2 + embed x1
            r3 = health.run_health(force=True, cache_ttl=30)
            self.assertEqual(calls["n"], 6)
        finally:
            health.check_model, health.check_embed = orig
            health._cache.clear()


if __name__ == "__main__":
    unittest.main()
