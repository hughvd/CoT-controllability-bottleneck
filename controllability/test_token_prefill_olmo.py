"""Regression checks for OLMo's native opening and multi-token reasoning close."""
import json
import unittest
import token_prefill_olmo as o

class OlmoAdapterTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tok=o.s.tokenizer(o.SPEC)
  cls.manifest=json.loads((o.ROOT/'olmo31/manifest.json').read_text())
 def test_template_opens_thinking_once(self):
  text=self.tok.decode(o.opening(self.tok,'A test question.'))
  self.assertTrue(text.endswith('<think>'))
  self.assertEqual(text.count('<think>'),1)
 def test_multitoken_close_excludes_final_answer(self):
  for prefix in ['', 'HELLO', 'Okay,']:
   for reason in ['', ' STILL THINKING', 'REASONING.\n']:
    for separator in ['', '\n', ' ']:
     with self.subTest(prefix=prefix,reason=reason,separator=separator):
      text=reason+separator+'</think>\nANSWER'
      ids=self.tok.encode(text,add_special_tokens=False)
      request={'prefix_token_ids':self.tok.encode(prefix,add_special_tokens=False)}
      full,head,n,closed=o.reasoning_parts(self.tok,request,{'output_token_ids':ids},self.manifest)
      self.assertEqual(full,reason+separator);self.assertTrue(closed)
      self.assertEqual(head,full)
      self.assertLess(n,len(ids))
 def test_eos_without_reasoning_close(self):
  for text in ['', 'KEEP THINKING']:
   ids=self.tok.encode(text,add_special_tokens=False)+[self.tok.eos_token_id]
   full,_,n,closed=o.reasoning_parts(self.tok,{'prefix_token_ids':[]},{'output_token_ids':ids},self.manifest)
   self.assertEqual(full,text);self.assertFalse(closed);self.assertEqual(n,len(ids)-1)
 def test_immediate_window_is_first_32_native_tokens(self):
  ids=self.tok.encode('REASONING. '*80+'</think>ANSWER',add_special_tokens=False)
  full,head,_,_=o.reasoning_parts(self.tok,{'prefix_token_ids':[]},{'output_token_ids':ids},self.manifest)
  self.assertEqual(head,o.s.base.decode(self.tok,ids[:32]));self.assertTrue(full.startswith(head))
 def test_manifest_matches_parent_and_retains_native_prefixes(self):
  m=self.manifest;o.s.validate(m)
  self.assertEqual(m['adapter_sha256'],o.s.base.file_hash(o.__file__))
  parent=json.loads((o.s.HERE/'results/token_prefill/full_v2/glm/manifest.json').read_text())
  self.assertEqual(m['parent_manifest_sha256'],parent['manifest_sha256'])
  for cell,source in zip(m['cells'],parent['cells']):
   self.assertEqual({k:v for k,v in cell.items() if k!='controlled_token_ids'},
                    {k:v for k,v in source.items() if k!='controlled_token_ids'})
  cells={c['cell_id']:c for c in m['cells']}
  for r in m['requests']:
   self.assertEqual(r['prefix_token_ids'],cells[r['cell_id']]['controlled_token_ids'][:r['prefix_tokens']])
   self.assertEqual(r['prompt_token_ids'],r['opening_token_ids']+r['prefix_token_ids'])
  self.assertEqual(len(m['links']),4800)
  self.assertEqual(len(m['requests']),4784)

if __name__=='__main__':unittest.main()
