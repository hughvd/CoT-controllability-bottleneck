"""Offline tests: python -m unittest discover -s controllability/mech -p test_rank_probe.py"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import rank_probe as rp


class Tokenizer:
    pieces = {0: "O", 1: "kay", 2: ",", 3: "OKAY", 4: "0k4y", 5: "</think>", 6: "abc"}

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[i] for i in ids)


class RankTests(unittest.TestCase):
    def test_instruction_removal_exact(self):
        s = "USER\n<cot_instructions> UPPERCASE </cot_instructions>\nQuestion: X\nANSWER: X\nASSISTANT<think>\n"
        self.assertEqual(rp.remove_instruction(s), "USER\nQuestion: X\nANSWER: X\nASSISTANT<think>\n")
        self.assertEqual(rp.remove_instruction(s.replace("UPPERCASE", "a->4\ne->3")), rp.remove_instruction(s))
        for bad in ["none", s + s, s.replace("</cot_instructions>\n", "</cot_instructions>")]:
            with self.assertRaises(ValueError):
                rp.remove_instruction(bad)

    def test_first_violation_split_tokens(self):
        t = Tokenizer()
        self.assertEqual(rp.first_violation(t, [0, 1, 2], "uppercase_thinking")["token_position"], 1)
        self.assertEqual(rp.first_violation(t, [0, 1, 2], "leetspeak")["token_position"], 0)
        self.assertIsNone(rp.first_violation(t, [3, 2, 5, 6], "uppercase_thinking"))
        self.assertIsNone(rp.first_violation(t, [4, 2], "leetspeak"))

    def test_masks(self):
        txt = ["OKAY", "Okay", "0k4y", " ", ",", "123", "Я", "я", "\ufffd", "<think>", "", "B", "\ud800", "中"]
        upper = rp.masks_from_texts(txt, {9}, "uppercase_thinking")
        leet = rp.masks_from_texts(txt, {9}, "leetspeak")
        self.assertEqual(np.flatnonzero(upper["content"]).tolist(), [0, 6, 11])
        self.assertEqual(np.flatnonzero(leet["content"]).tolist(), [2, 5, 11])
        self.assertEqual(np.flatnonzero(upper["neutral"]).tolist(), [3, 4])
        self.assertEqual(np.flatnonzero(leet["excluded"]).tolist(), [8, 9, 10, 12])
        self.assertTrue(upper["legal"][5])
        self.assertFalse(upper["content"][5])

    def test_tied_ranks_and_full_vocabulary(self):
        txt = ["bad", " ", "UP", "YES", "B"]
        z = np.array([8., 7., 5., 5., -12.], np.float32)
        m = rp.masks_from_texts(txt, set(), "uppercase_thinking")
        s, lp = rp.summarize(z, m, txt, [0, 4])
        self.assertEqual(s["best_content"]["rank"], 3)
        self.assertEqual(rp.token_stat(z, lp, 3, txt)["rank"], 3)
        self.assertEqual(s["observed"]["4"]["rank"], 5)
        # Rank search is not capped at the saved top-200.
        z = np.arange(400, 0, -1, dtype=np.float32)
        txt = ["lower"] * 399 + ["UP"]
        s, _ = rp.summarize(z, rp.masks_from_texts(txt, set(), "uppercase_thinking"), txt, [399])
        self.assertEqual(s["best_content"]["rank"], 400)
        self.assertEqual(len(s["top200"]), 200)
        self.assertIsNotNone(s["content_log_mass"])

    def test_nucleus_crossing(self):
        z = np.log(np.array([0.6, 0.3, 0.1], np.float32))
        p, keep = rp.nucleus(z, temperature=1, top_p=0.8)
        np.testing.assert_array_equal(keep, [True, True, False])
        np.testing.assert_allclose(p, [2/3, 1/3, 0], atol=1e-7)
        _, keep = rp.nucleus(z, temperature=1, top_p=0.95)
        self.assertTrue(keep.all())

    def test_fixed_candidates_and_probability_mass(self):
        txt = ["Okay", "OKAY", "YES", " "]
        z0 = np.array([6, 2, 1, 4], np.float32)
        z1 = np.array([5, 3, 4, 2], np.float32)
        m = rp.masks_from_texts(txt, set(), "uppercase_thinking")
        out = rp.paired_statistics(dict(zip(rp.CONDS, [z0, z1])), m, txt, [0])
        self.assertEqual(set(out["fixed_candidates"]), {"1", "2"})
        self.assertGreater(out["conditions"]["instruction_only"]["content_mass"], out["conditions"]["no_instruction"]["content_mass"])
        self.assertGreater(out["distribution_shift"]["total_variation"], 0)
        same = rp.paired_statistics(dict(zip(rp.CONDS, [z0, z0])), m, txt, [0])
        self.assertEqual(same["distribution_shift"]["total_variation"], 0)
        from analysis.rank_analysis import distribution_summary, flatten
        row = dict(pair_id="test", model_tag="test", ctype="uppercase_thinking", prompt_id="p",
                   question_id="q", position=0, references=[{}], **out)
        shifts = distribution_summary(flatten([row])).set_index("metric")
        self.assertAlmostEqual(shifts.loc["total_variation", "mean"], out["distribution_shift"]["total_variation"])

    def test_manifest_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "manifest.json"
            m = {"version": rp.VERSION, "pairs": []}
            m["manifest_sha256"] = rp.digest(m)
            rp.immutable_json(path, m)
            rp.immutable_json(path, m)
            self.assertEqual(rp.load_manifest(path), m)
            with self.assertRaises(ValueError):
                rp.immutable_json(path, {"changed": True})
            fp = Path(d) / "results.jsonl"
            r = {"version": rp.VERSION, "manifest_sha256": m["manifest_sha256"], "pair_id": "one"}
            fp.write_bytes((rp.canonical(r) + b"\n") * 2)
            self.assertEqual(len(rp.load_results(fp, m["manifest_sha256"])), 1)
            with fp.open("ab") as f:
                f.write(b'{"incomplete":')
            self.assertEqual(len(rp.load_results(fp, m["manifest_sha256"])), 1)
            with self.assertRaises(ValueError):
                rp.load_results(fp, "wrong")

    def test_question_cluster_weighting(self):
        from analysis.rank_analysis import question_summary
        import pandas as pd
        # One question has two different saved prefixes, another only one.
        # References to repeated samples must never become bootstrap rows.
        metrics = ["log10_rank", "best_log10_probability", "content_log10_mass", "content_mass",
                   "legal_mass", "content_sample_mass", "legal_sample_mass", "neutral_mass",
                   "excluded_mass", "entropy_nats", "argmax_probability", "total_variation"]
        rows = []
        for question, changes in [("q1", [1., 3.]), ("q2", [0.])]:
            for change in changes:
                for cond in rp.CONDS:
                    rows.append(dict(model_tag="test", ctype="test", position=0, question_id=question,
                                     condition=cond, n_source_traces=20,
                                     **{m: change if cond == rp.CONDS[1] else 0. for m in metrics}))
        q, summary = question_summary(pd.DataFrame(rows))
        self.assertEqual(len(q), 4)
        self.assertTrue((summary.n_questions == 2).all())
        self.assertTrue((summary.delta == 1.).all())

    def test_prepared_manifest_prefixes(self):
        path = rp.DATA / "rank_runs/opening_ranks_v1/manifest.json"
        if not path.exists():
            self.skipTest("Primary data has not been prepared on this machine")
        manifest = rp.load_manifest(path)
        self.assertEqual(len(manifest["traces"]), 144)
        pairs = {p["pair_id"]: p for pool in manifest["pairs"].values() for p in pool}
        for trace in manifest["traces"]:
            self.assertEqual(trace["violation"]["token_position"], 0)
            for target in trace["targets"]:
                pair = pairs[target["pair_id"]]
                t = target["position"]
                self.assertEqual(pair["prefix_ids"], trace["output_token_ids"][:t])
                self.assertIn(trace["output_token_ids"][t], pair["observed_ids"])
        for pool in manifest["pairs"].values():
            openings = [p for p in pool if p["position"] == 0]
            self.assertEqual(len(openings), 48)
            self.assertEqual(len({p["prompt_id"] for p in openings}), 48)


if __name__ == "__main__":
    unittest.main()
