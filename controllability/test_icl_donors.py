import random
import unittest
from dataclasses import replace
import icl_experiment as i

class DonorTests(unittest.TestCase):
    def test_cross_behavior_question_collision(self):
        cells = [i.Cell(ct,'family',q,[], 'A','EXAMPLE','control '+ct)
                 for ct in ['uppercase_thinking','leetspeak'] for q in ['same question','other question']]
        donors=i.select_donors(cells,0)
        self.assertEqual(len(donors),4)
        for c in cells:
            d=donors[c.ctype,c.question]
            self.assertEqual(d.ctype,c.ctype)
            self.assertNotEqual(d.question,c.question)
        with self.assertRaises(ValueError):
            i.build_icl_messages(cells[0],cells[3],'instruction_only',None)
        with self.assertRaises(ValueError):
            i.grade_cell_for(cells[0],cells[3])

    def test_actual_sweep_preserves_unaffected_assignments(self):
        from icl_repair import assignments
        cells,old,fixed=assignments()
        affected=0
        for c in cells:
            d=fixed[c.ctype,c.question]
            self.assertEqual(d.ctype,c.ctype)
            if old[c.question].ctype==c.ctype:self.assertEqual(d,old[c.question])
            else:affected+=1
        self.assertEqual(affected,16)


class HarmonyTests(unittest.TestCase):
    def test_visible_example_and_unseeded_completion(self):
        import icl_harmony as ih
        import harmony_backend as h
        messages=[{'role':'user','content':'example question'},
                  {'role':'assistant','content':'EXAMPLE REASONING. ANSWER: B'},
                  {'role':'user','content':'new question'}]
        text=h._enc().decode(ih.build_tokens(messages))
        self.assertIn('Reasoning: high',text)
        self.assertIn('<|start|>assistant<|channel|>final<|message|>EXAMPLE REASONING.',text)
        self.assertTrue(text.endswith('<|start|>assistant'))
        self.assertNotIn('<|channel|>analysis',text)

class ResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_generation_is_reused_after_grader_interruption(self):
        import asyncio
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch
        from openai.types.chat import ChatCompletion
        class Cache:
            def __init__(self):self.data={}
            def __contains__(self,key):return key in self.data
            async def set(self,key,value):self.data[key]=value
        raw={'id':'test','object':'chat.completion','created':0,'model':'test',
             'choices':[{'index':0,'finish_reason':'stop','message':
                         {'role':'assistant','content':'ANSWER: A','reasoning':'THINKING.'}}]}
        response=i.pe._parse_response(ChatCompletion.model_validate(raw))
        cfg=SimpleNamespace(run_name='test',save_raw=True,model_short='test')
        a=i.Cell('uppercase_thinking','family','target',[],'A','SOURCE.','uppercase')
        b=replace(a,question='donor')
        cache=Cache();complete=AsyncMock(return_value=(response,raw))
        with tempfile.TemporaryDirectory() as tmp, patch.object(i,'TRANSCRIPTS',Path(tmp)), patch.object(i,'complete_pinned',complete):
            with patch.object(i,'grade_continuation',AsyncMock(side_effect=RuntimeError('interrupted'))):
                with self.assertRaises(RuntimeError):
                    await i.run_one_icl(a,b,'instruction_only',None,None,0,cfg,cache,asyncio.Semaphore(1))
            with patch.object(i,'grade_continuation',AsyncMock(return_value=(1.,1))):
                await i.run_one_icl(a,b,'instruction_only',None,None,0,cfg,cache,asyncio.Semaphore(1))
            self.assertEqual(complete.await_count,1)
            row=next(iter(cache.data.values()))
            self.assertEqual(row['reasoning'],'THINKING.')
            self.assertEqual(row['donor_ctype'],row['ctype'])

class NativeProtocolTests(unittest.TestCase):
    def test_olmo_reasoning_and_truncation(self):
        from olmo_icl_protocol import split_reasoning
        self.assertEqual(split_reasoning(' THINKING. </think> ANSWER: B<|endoftext|>'),('THINKING.','ANSWER: B'))
        self.assertEqual(split_reasoning('unfinished thinking'),('unfinished thinking',''))

    def test_gpt_stop_token_text_difference(self):
        from gptoss_icl_run import validate_output_text
        import harmony_backend as h
        text='hello'
        validate_output_text({'raw_text':text,'output_token_ids':h._enc().encode(text+'<|return|>',allowed_special='all')})
        with self.assertRaises(AssertionError):
            validate_output_text({'raw_text':'different','output_token_ids':h._enc().encode(text)})

    def test_judge_apostrophe_repair(self):
        import json
        from icl_judging import repair_apostrophes
        bad='{"score": 0.25, "reason": "Meesa thinkin'+chr(92)+"'"+'"}'
        self.assertEqual(json.loads(repair_apostrophes(bad))['score'],.25)
        valid=json.dumps({'score':.5,'reason':"a \\ path and an apostrophe '"})
        self.assertEqual(repair_apostrophes(valid),valid)

if __name__=='__main__':unittest.main()
