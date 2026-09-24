import json
import tempfile
import unittest
from pathlib import Path
import gptoss_prefill_run as run
import harmony_backend as h


class PrefillRunTests(unittest.TestCase):
    def test_real_harmony_prefix_conditions(self):
        enc=h._enc()
        for effort in ['medium','high']:
            plain=enc.decode(h.build_prefill_tokens('Question?', ['A','B'], '', '', effort))
            seeded=enc.decode(h.build_prefill_tokens('Question?', ['A','B'], 'UPPERCASE', 'THE OPTIONS ARE', effort))
            self.assertNotIn('<|start|>developer',plain)
            self.assertTrue(plain.endswith('<|start|>assistant'))
            self.assertIn('Reasoning: '+effort,plain)
            self.assertTrue(seeded.endswith('<|channel|>analysis<|message|>THE OPTIONS ARE'))

    def test_torn_journal_tail_and_corrupt_middle(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'records.jsonl'
            path.write_text('{"key":"a"}\n{"key":')
            self.assertEqual(run.read_journal(path),[{'key':'a'}])
            with path.open('a') as f:f.write('{"key":"b"}\n')
            self.assertEqual(len(run.read_journal(path)),2)
            path.write_text('{broken}\n{"key":"b"}\n')
            with self.assertRaises(json.JSONDecodeError):run.read_journal(path)

    def test_manifest_preserves_source_matrix(self):
        m=json.loads((run.ROOT/'manifest.json').read_text())
        self.assertEqual(m['manifest_sha256'],run.digest({k:v for k,v in m.items() if k!='manifest_sha256'}))
        rows=m['requests'];self.assertEqual(len(rows),3600)
        self.assertEqual(len({r['request_id'] for r in rows}),3600)
        self.assertEqual(len({r['cell_id'] for r in rows}),300)
        self.assertTrue(all(len(r['prompt_token_ids'])+8192<=16384 for r in rows))
        self.assertEqual(len(m['smoke_ids']),120)


if __name__=='__main__':unittest.main()
