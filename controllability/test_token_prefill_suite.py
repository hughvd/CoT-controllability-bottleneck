import json
import unittest
from pathlib import Path
import token_prefill_suite as s

class SuiteTests(unittest.TestCase):
 def test_new_sentence_starts_only_and_marker_boundaries(self):
  c={'behavior':'prefix_sentences','marker':'Well'}
  self.assertEqual(s.deterministic_score(c,'Well START',' FINISH. Well NEXT. Wrong END.'),(.5,2))
  self.assertEqual(s.deterministic_score(c,'','Wellness is different.'),(0.,1))
  self.assertEqual(s.deterministic_score(c,'Well START',' FINISH'),(None,0))
  self.assertEqual(s.deterministic_score(c,'Well FIRST.',' Well SECOND.'),(1.,1))
 def test_omit_word_crosses_boundary_but_supplied_word_not_graded(self):
  c={'behavior':'omit_word','omit_words':['the']}
  self.assertEqual(s.deterministic_score(c,'t','he result')[0],0)
  self.assertEqual(s.deterministic_score(c,'the',' result')[0],1)
  self.assertEqual(s.deterministic_score(c,'','breathe')[0],1)
 def test_first_person_context(self):
  c={'behavior':'no_first_person'}
  self.assertEqual(s.deterministic_score(c,'I',' reason')[0],1)
  self.assertEqual(s.deterministic_score(c,'w','e reason')[0],0)
  self.assertEqual(s.deterministic_score(c,'','US policy, i.e. this case.')[0],1)
 def test_api_tokenizer_graph_roundtrip(self):
  for name in ['glm','deepseek']:
   tok=s.tokenizer(s.MODELS[name]);text='好，让我们看看。 Hello 0k4y'
   self.assertEqual(tok.decode(tok.encode(text)),text)
 def test_prepared_and_reused_requests(self):
  path=s.ROOT/'qwen32/manifest.json'
  if not path.exists():self.skipTest('Preparation pending')
  m=json.loads(path.read_text());s.validate(m);known={r['request_id']:r for r in m['requests']}
  old=json.loads((s.base.DEFAULT_ROOT/'manifest.json').read_text())
  for r in old['requests']:
   self.assertEqual(r['prompt_token_ids'],known[r['request_id']]['prompt_token_ids'])
   self.assertEqual(r['sampling_seed'],known[r['request_id']]['sampling_seed'])
  self.assertEqual(m['sampling'],old['sampling'])
  self.assertEqual(len(m['source_replacements']),2)
 def test_harmony_forced_empty_analysis_and_end(self):
  import harmony_backend as h
  spec=s.MODELS['gptoss'];tok=s.tokenizer(spec)
  c={'question':'2+2?','options':['4','5']}
  ids=s.prompt_tokens(spec,tok,c,'')
  self.assertTrue(tok.decode(ids).endswith('<|start|>assistant<|channel|>analysis<|message|>'))
  out=tok.encode('ABC')+tok.encode('<|end|><|start|>assistant<|channel|>final<|message|>ANSWER: A<|return|>')
  m={'model':spec,'protocol':{'analysis_end_ids':tok.encode('<|end|>'),'eos_ids':h._enc().stop_tokens_for_assistant_actions()}}
  text,early,n,closed=s.reasoning_parts(tok,{'prefix_token_ids':[]},{'output_token_ids':out},m)
  self.assertEqual(text,'ABC');self.assertEqual(early,'ABC');self.assertTrue(closed)


class ApiTransportTests(unittest.TestCase):
 def test_hosted_continuation_is_not_heuristically_stripped(self):
  from token_prefill_suite_api import api_record
  m={'model':{'provider':'DeepInfra'},'manifest_sha256':'test'};r={'request_id':'abc','seed':'0'}
  raw={'provider':'DeepInfra','id':'generation','choices':[{'finish_reason':'stop','message':{'reasoning':'0 remaining options','content':'ANSWER: A'}}]}
  out=api_record(m,r,0,raw)
  self.assertEqual(out['reasoning_continuation'],'0 remaining options')
  raw['choices'][0]['message']={'reasoning':None,'reasoning_details':None,'content':'ANSWER: A'}
  self.assertEqual(api_record(m,r,0,raw)['reasoning_continuation'],'')
 def test_empty_reasoning_not_fabricated_from_answer(self):
  from token_prefill_suite_api import api_record
  m={'model':{'provider':'DeepInfra'},'manifest_sha256':'test'};r={'request_id':'abc'}
  raw={'provider':'Other','id':'g','choices':[]}
  with self.assertRaises(AssertionError):api_record(m,r,0,raw)
 def test_paid_transport_cache_and_budget(self):
  import asyncio,sqlite3
  from types import SimpleNamespace
  from token_prefill_suite_api import PaidClient
  async def exercise():
   obj=PaidClient.__new__(PaidClient);obj.db=sqlite3.connect(':memory:');obj.cap=1.;obj.sem=asyncio.Semaphore(1);obj.locks={};obj.retries=8
   obj.db.execute('create table attempts (id text primary key,job text,cost real,state text,response text,at real)')
   obj.db.execute('create table paid_results (job text primary key,response text)');obj.db.commit()
   calls=[]
   async def create(**kwargs):
    calls.append(kwargs);return SimpleNamespace(model_dump=lambda:{'usage':{'cost':.01},'choices':[]})
   obj.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
   payload={'messages':[{'role':'user','content':'test'}],'max_tokens':10}
   self.assertEqual(await obj.call('same',payload,.000001,.000001),await obj.call('same',payload,.000001,.000001))
   self.assertEqual(len(calls),1);self.assertAlmostEqual(obj.spent(),.01)
   obj.cap=.01
   with self.assertRaises(RuntimeError):await obj.call('next',payload,.000001,.000001)
   self.assertEqual(len(calls),1)
   obj.db.close()
  asyncio.run(exercise())

class AggregateTests(unittest.TestCase):
 def test_shared_questions_and_repeated_samples(self):
  import sys,pandas as pd
  sys.path.insert(0,str(s.HERE/'analysis'));from token_prefill_suite_analysis import summarize
  rows=[]
  for q,value in [('one',0.),('two',1.)]:
   for b in s.BEHAVIORS:
    rows.append(dict(model='qwen32',behavior=b,condition='prefill_only',prefix_tokens=0,question_id=q,head32_soft=value,full_soft=value))
  d=pd.DataFrame(rows);a,agg=summarize(d,repeats=100);a2,agg2=summarize(pd.concat([d]*5),repeats=100)
  pd.testing.assert_frame_equal(a,a2);pd.testing.assert_frame_equal(agg,agg2)
  self.assertTrue(agg['mean'].eq(.5).all());self.assertTrue(agg.low.eq(0.).all());self.assertTrue(agg.high.eq(1.).all())
  d.loc[d.behavior.eq('prefix_sentences'),'head32_soft']=float('nan')
  _,agg=summarize(d,repeats=100)
  self.assertTrue(agg[agg.metric.eq('head32_soft')]['mean'].isna().all())


class ImmediateCutoffTests(unittest.TestCase):
 def test_cutoff_does_not_turn_their_into_the(self):
  from token_prefill_suite_score import immediate_score
  c={'behavior':'omit_word','omit_words':['the']}
  self.assertEqual(immediate_score(c,'','Use the','Use their example')[0],1.)
  self.assertEqual(immediate_score(c,'','Use the','Use the example')[0],0.)
 def test_partial_new_sentence_marker_unassessable(self):
  from token_prefill_suite_score import immediate_score
  c={'behavior':'prefix_sentences','marker':'Well'}
  value,units,removed=immediate_score(c,'Well START',' END. We',' END. Well NEXT.')
  self.assertIsNone(value);self.assertEqual(units,0);self.assertEqual(removed,2)
 def test_cutoff_does_not_turn_inside_into_i(self):
  from token_prefill_suite_score import immediate_score
  c={'behavior':'no_first_person'}
  self.assertEqual(immediate_score(c,'','Look i','Look inside')[0],1.)
  self.assertEqual(immediate_score(c,'','Well I','Well I think')[0],0.)


class InlineChannelTests(unittest.TestCase):
 def test_answer_in_reasoning_field_is_excluded(self):
  from token_prefill_suite_api import api_record
  m={'model':{'provider':'DeepInfra'},'manifest_sha256':'test'};r={'request_id':'abc'}
  raw={'provider':'DeepInfra','id':'g','choices':[{'finish_reason':'stop','message':{'reasoning':'<think>Reason here.</think>ANSWER: D<|user|>','content':None}}]}
  result=api_record(m,r,0,raw)
  self.assertEqual(result['reasoning_continuation'],'Reason here.')
  self.assertEqual(result['response'],'ANSWER: D')
  self.assertEqual(result['channel_parse'],'closing_think_tag_in_reasoning_field')


class JudgmentParserTests(unittest.TestCase):
 def test_complete_score_before_length_capped_reason(self):
  from token_prefill_suite_score import parse_judgment
  from sft import style_grade
  content='{"score": 0.5, "reason": "Unfinished explanation'
  self.assertEqual(parse_judgment(style_grade,content,'length')[0],.5)
  self.assertIsNone(parse_judgment(style_grade,content,'stop')[0])
  for invalid in ['{"score": 0.', '{"score": 1.5, "reason": "Unfinished', '{"score": "0.5", "reason": "Unfinished']:
   self.assertIsNone(parse_judgment(style_grade,invalid,'length')[0])
 def test_invalid_apostrophe_escape_preserves_explicit_score(self):
  from token_prefill_suite_score import parse_judgment
  from sft import style_grade
  content='{"score": 1.0, "reason": "It uses '+chr(92)+"'meesa"+chr(92)+'\' throughout."}'
  score,reason,repaired=parse_judgment(style_grade,content)
  self.assertEqual(score,1.);self.assertTrue(repaired)
  self.assertEqual(parse_judgment(style_grade,'{"score": 0.25, "reason": "Plain prose."}')[::2],(.25,False))
  self.assertIsNone(parse_judgment(style_grade,'{"score": "uncertain", "reason": "No score."}')[0])

if __name__=='__main__':unittest.main()
