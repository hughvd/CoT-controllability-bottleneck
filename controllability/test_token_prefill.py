import json
from pathlib import Path
import tempfile
import unittest

import token_prefill as tp


class ByteTokenizer:
    def decode(self, ids, **kwargs):
        chunks, buf = [], []
        for token in ids:
            if token >= 1000:
                chunks.append(bytes(buf).decode("utf-8", errors="replace"))
                chunks.append({1000: "</think>", 1001: "<eos>"}[token])
                buf = []
            else:
                buf.append(token)
        chunks.append(bytes(buf).decode("utf-8", errors="replace"))
        return "".join(chunks)


class TokenPrefillTests(unittest.TestCase):
    def test_scores_match_sentence_grader_and_flag_vacuity(self):
        from transforms import grade_score
        for behavior in tp.BEHAVIORS:
            for text in ["", "  ", "HELLO", "Hello", "123...", "Ωω中文", "4 3 1 0 7 5", "aeiots", "LET'S GO"]:
                with self.subTest(behavior=behavior, text=text):
                    expected = grade_score(text, behavior) if text.strip() else 0.
                    self.assertEqual(tp.score_text(text, behavior)["soft"], expected)
        self.assertEqual(tp.score_text("...", "uppercase_thinking")["soft"], 1)
        self.assertEqual(tp.score_text("...", "uppercase_thinking")["content_soft"], 0)
        self.assertEqual(tp.score_text("...", "uppercase_thinking")["strict"], 0)

    def test_seed_and_answer_excluded_head32_and_failure_index(self):
        tok = ByteTokenizer()
        request = {"prefix_token_ids": list(b"lowercase seed ")}
        output = list(b"A" * 32 + b"b") + [1000] + list(b"ANSWER: E") + [1001]
        result = tp.measure(tok, request, output, {"think_close_id": 1000, "eos_ids": [1001]}, "uppercase_thinking")
        self.assertEqual(result["head32_soft"], 1.)
        self.assertAlmostEqual(result["full_soft"], 32 / 33)
        self.assertEqual(result["reasoning_tokens"], 33)
        self.assertEqual(result["first_violation_token"], 32)
        self.assertTrue(result["closed_reasoning"])
        self.assertNotIn("seed", result["reasoning_text"])

    def test_cut_word_diagnostic_and_short_empty_reasoning(self):
        tok = ByteTokenizer()
        protocol = {"think_close_id": 1000, "eos_ids": [1001]}
        r = {"prefix_token_ids": list(b"O")}
        result = tp.measure(tok, r, list(b"KAY then") + [1000], protocol, "uppercase_thinking")
        self.assertGreater(result["head32_soft"], 0)
        self.assertEqual(result["head32_after_boundary_content_soft"], 0)
        self.assertFalse(result["has_32_tokens"])
        self.assertTrue(tp.measure(tok, r, [1000], protocol, "uppercase_thinking")["empty_reasoning"])
        self.assertEqual(tp.after_boundary(" A NEW WORD", "END"), " A NEW WORD")

    def test_unicode_boundary_joint_decoding(self):
        tok = ByteTokenizer()
        self.assertEqual(tp.continuation_text(tok, [0xCE], [0xA9, 32, 65]), "Ω A")
        self.assertEqual(tp.continuation_text(tok, [0xCE], []), "")
        r = tp.measure(tok, {"prefix_token_ids": [0xCE]}, [0xA9, 32, 97, 1000],
                       {"think_close_id": 1000, "eos_ids": [1001]}, "uppercase_thinking")
        self.assertTrue(r["seed_incomplete_utf8"])
        self.assertEqual(r["first_violation_token"], 2)

    def test_immutable_artifact_and_tampered_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            tp.immutable_json(path, {"a": [1, 2]})
            tp.immutable_json(path, {"a": [1, 2]})
            with self.assertRaises(ValueError):
                tp.immutable_json(path, {"a": [1, 3]})
        with self.assertRaises(AssertionError):
            tp.validate_manifest({"manifest_sha256": "not-the-content-hash"})

    def test_real_qwen_prefix_ids_are_nested_and_not_retokenized(self):
        from transformers import AutoTokenizer
        path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots" / tp.REVISION
        if not path.exists():
            self.skipTest("Pinned Qwen tokenizer not cached")
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        seed = "OKAY, LET'S REASON THROUGH THE QUESTION. " * 12
        seed_ids = tok.encode(seed, add_special_tokens=False)
        text = tok.apply_chat_template([{"role": "user", "content": "Question?"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True) + "<think>\n"
        opening = tok.encode(text, add_special_tokens=False)
        for length in tp.LENGTHS:
            prompt = opening + seed_ids[:length]
            self.assertEqual(prompt[:len(opening)], opening)
            self.assertEqual(len(prompt) - len(opening), length)
            if length:
                self.assertEqual(tp.decode(tok, prompt), text + tp.decode(tok, seed_ids[:length]))

    def test_bootstrap_does_not_count_repeated_samples_as_questions(self):
        import sys
        import pandas as pd
        sys.path.insert(0, str(Path(tp.__file__).parent / "analysis"))
        import token_prefill_analysis as analysis
        rows = []
        for question, score in [("one", 0.), ("two", 1.)]:
            for condition in tp.CONDITIONS:
                for length in [0, 1]:
                    rows.append(dict(question_id=question, behavior="uppercase_thinking", condition=condition,
                                     prefix_tokens=length, **{metric: score for metric in analysis.METRICS}))
        d = pd.DataFrame(rows)
        one = analysis.clustered_summary(d, repeats=100)
        repeated = analysis.clustered_summary(pd.concat([d] * 5), repeats=100)
        pd.testing.assert_frame_equal(one, repeated)
        self.assertTrue(one.questions.eq(2).all())

    def test_scoring_rejects_partial_and_duplicate_batches(self):
        path = tp.DEFAULT_ROOT / "manifest.json"
        if not path.exists():
            self.skipTest("Run prepare first for the local pipeline fixture")
        m = json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tp.atomic_json(root / "manifest.json", m)
            with self.assertRaisesRegex(ValueError, "Incomplete outputs"):
                tp.score(root)
            r = m["requests"][0]
            row = dict(request_id=r["request_id"], sample=0, prompt_token_ids=r["prompt_token_ids"],
                       output_token_ids=[], finish_reason="stop")
            batch = dict(manifest_sha256=m["manifest_sha256"], outputs=[row])
            tp.atomic_json(root / "remote/batch_one.json", batch)
            tp.atomic_json(root / "remote/batch_duplicate.json", batch)
            with self.assertRaises(AssertionError):
                tp.score(root)
            self.assertFalse((root / "scores.csv").exists())

    def test_paired_intervals_cancel_between_question_baselines(self):
        import sys
        import pandas as pd
        sys.path.insert(0, str(Path(tp.__file__).parent / "analysis"))
        import token_prefill_analysis as analysis
        rows = []
        for question, baseline in [("one", .1), ("two", .6)]:
            for condition in tp.CONDITIONS:
                for length in [0, 1]:
                    score = baseline + .2 * length + .1 * (condition == "prefill_plus_instruction")
                    rows.append(dict(question_id=question, behavior="uppercase_thinking", condition=condition,
                                     prefix_tokens=length, **{metric: score for metric in analysis.METRICS}))
        d = pd.DataFrame(rows)
        result = analysis.paired_changes(d, repeats=100)
        pd.testing.assert_frame_equal(result, analysis.paired_changes(pd.concat([d] * 5), repeats=100))
        for row in result.itertuples():
            expected = .2 * row.prefix_tokens if row.contrast == "prefix_minus_zero" else .1
            for value in [row.mean, row.low, row.high]:
                self.assertAlmostEqual(value, expected)

    def test_synthetic_pipeline_scores_resumably_and_renders(self):
        from unittest.mock import patch
        from transformers import AutoTokenizer
        import sys
        import pandas as pd
        path = tp.DEFAULT_ROOT / "manifest.json"
        if not path.exists():
            self.skipTest("Run prepare first for the local pipeline fixture")
        m = json.loads(path.read_text())
        m["cells"] = [next(c for c in m["cells"] if c["behavior"] == b) for b in tp.BEHAVIORS]
        selected = {c["cell_id"] for c in m["cells"]}
        m["links"] = [x for x in m["links"] if x["cell_id"] in selected]
        needed = {x["request_id"] for x in m["links"]}
        m["requests"] = [r for r in m["requests"] if r["request_id"] in needed]
        m["run"] = "synthetic_test_only"
        m["manifest_sha256"] = tp.digest({k: v for k, v in m.items() if k != "manifest_sha256"})
        tok = AutoTokenizer.from_pretrained(tp.MODEL, revision=tp.REVISION, local_files_only=True)
        ids = tok.encode("ABC def", add_special_tokens=False) + [m["protocol"]["think_close_id"]]
        output = [dict(request_id=r["request_id"], sample=s, prompt_token_ids=r["prompt_token_ids"],
                       output_token_ids=ids, finish_reason="stop")
                  for r in m["requests"] for s in range(m["sampling"]["n"])]
        sys.path.insert(0, str(Path(tp.__file__).parent / "analysis"))
        import token_prefill_analysis as analysis
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tp.atomic_json(root / "manifest.json", m)
            tp.atomic_json(root / "remote/batch_synthetic.json",
                           dict(manifest_sha256=m["manifest_sha256"], outputs=output))
            tp.score(root)
            original = (root / "scores.csv").read_bytes()
            tp.score(root)
            self.assertEqual(original, (root / "scores.csv").read_bytes())
            self.assertEqual(len(pd.read_csv(root / "scores.csv")), 160)
            # Keep test exports out of the real manuscript's figures folder.
            import paper_style
            with patch.object(analysis, "HERE", root / "analysis"), patch.object(paper_style, "PAPER_FIGURES", root / "paper_figures"):
                analysis.main(root)
            self.assertTrue((root / "analysis/token_prefill/synthetic_test_only/compliance.png").exists())


if __name__ == "__main__":
    unittest.main()
