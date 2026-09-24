import unittest
import numpy as np
import suite_rank_probe as s


class FakeTokenizer:
    all_special_ids = [9]
    pieces = {0: 'O', 1: 'kay', 2: ',', 3: ' 12', 4: '. Next', 5: ' B', 6: 'EEP', 9: '</think>'}
    def decode(self, ids, **kwargs):
        return ''.join(self.pieces[i] for i in ids)


class SuiteTests(unittest.TestCase):
    def test_disemvowel_word_state(self):
        self.assertIsNone(s.scan('A nthr wrd. I knw', 'disemvowel')[1])
        self.assertEqual(s.scan('Okay', 'disemvowel')[1]['character_position'], 2)
        state = s.scan('O', 'disemvowel')[0]
        self.assertIsNotNone(s.scan('a', 'disemvowel', state=state)[1])
        self.assertIsNone(s.scan(' a', 'disemvowel', state=state)[1])

    def test_suppression_unicode(self):
        self.assertIsNotNone(s.scan('５', 'spell_out_numbers')[1])
        self.assertIsNone(s.scan('² five', 'spell_out_numbers')[1])  # regex \\d semantics
        self.assertIsNotNone(s.scan('Okay,', 'no_commas')[1])
        self.assertIsNone(s.scan('，', 'no_commas')[1])

    def test_prefix_progress_boundary(self):
        a, bad, progress, complete = s.scan('  BE', 'start_with', 'BEEP')
        self.assertIsNone(bad)
        self.assertEqual((progress, complete), (2, 0))
        b, bad, progress, complete = s.scan('EP rest', 'start_with', 'BEEP', a)
        self.assertIsNone(bad)
        self.assertEqual((progress, complete), (2, 1))
        self.assertFalse(s.constraint_active('start_with', b))
        self.assertIsNotNone(s.scan('BEEPER', 'start_with', 'BEEP')[1])
        self.assertIsNotNone(s.scan('BEEP. Next', 'start_with', 'BEEP')[1])

    def test_sentence_prefixes(self):
        self.assertIsNone(s.scan('BEEP Hi.\nBEEP More!', 'prefix_sentences', 'BEEP')[1])
        bad = s.scan('BEEP Hi.\nWrong', 'prefix_sentences', 'BEEP')[1]
        self.assertEqual(bad['sentence_index'], 2)
        self.assertEqual(bad['character'], 'W')
        state = s.scan('BEEP Hi.\n', 'prefix_sentences', 'BEEP')[0]
        self.assertEqual(state['phase'], 'header')
        self.assertIsNone(s.scan('BEEP More', 'prefix_sentences', 'BEEP', state)[1])

    def test_numbering_period_and_partial_tokens(self):
        text = '1. Hello. 2. Goodbye!\n3. Yes'
        state, bad, _, count = s.scan(text, 'numbered_sentences')
        self.assertIsNone(bad)
        self.assertEqual(state['sentence'], 3)
        self.assertEqual(count, 3)
        self.assertEqual(s.scan('1. Hello. 3. Wrong', 'numbered_sentences')[1]['sentence_index'], 2)
        state = s.scan('1', 'numbered_sentences')[0]
        self.assertIsNone(s.scan('. Hi', 'numbered_sentences', state=state)[1])

    def test_candidate_sets(self):
        txt = [' ', 'BE', 'BEEP', 'BEEP ', 'BEEPER', 'Okay', '<think>', '\ufffd']
        m = s.masks_from_state(txt, {6}, 'start_with', 'BEEP', s.initial_state())
        self.assertEqual(np.flatnonzero(m['content']).tolist(), [1, 2, 3])
        self.assertEqual(np.flatnonzero(m['completion']).tolist(), [3])
        self.assertEqual(np.flatnonzero(m['forbidden']).tolist(), [4, 5])
        self.assertEqual(np.flatnonzero(m['excluded']).tolist(), [6, 7])

    def test_violation_token_mapping(self):
        tok = FakeTokenizer()
        self.assertEqual(s.first_violation(tok, [0,1,2,9], 'no_commas')['token_position'], 2)
        self.assertEqual(s.first_violation(tok, [0,1,3,9], 'spell_out_numbers')['token_position'], 2)
        self.assertEqual(s.first_violation(tok, [0,1,9], 'disemvowel')['token_position'], 1)
        self.assertIsNone(s.first_violation(tok, [5,6,9], 'start_with', 'BEEP'))
        self.assertTrue(s.first_violation(tok, [5,9], 'start_with', 'BEEP')['termination'])

    def test_chunked_scanning_matches_whole_text(self):
        for ct, marker, text in [('prefix_sentences','BEEP','BEEP Hi. BEEP Yes! BEEP Sure'),
                                 ('numbered_sentences',None,'1. Hi. 2. Good. 3. Fine'),
                                 ('disemvowel',None,'A wrd. I thnk'),
                                 ('no_commas',None,'Okay, this is fine.')]:
            whole = s.scan(text,ct,marker)
            for cut in range(len(text)+1):
                a=s.scan(text[:cut],ct,marker)
                b=s.scan(text[cut:],ct,marker,a[0])
                self.assertEqual(whole[0], b[0])
                bad = a[1] or (dict(b[1],character_position=b[1]['character_position']+cut) if b[1] else None)
                self.assertEqual(whole[1],bad)

    def test_question_weighting_and_no_failure(self):
        import pandas as pd
        from analysis.suite_rank_analysis import failure_summary, FAILURE_LABELS
        rows=[]
        for question,prompt,categories in [('q1','a',[FAILURE_LABELS[0],FAILURE_LABELS[3]]),
                                           ('q1','b',[FAILURE_LABELS[0],FAILURE_LABELS[0]]),
                                           ('q2','c',[FAILURE_LABELS[3],FAILURE_LABELS[3]])]:
            rows += [dict(model_tag='test',ctype='test',question_id=question,prompt_id=prompt,category=c) for c in categories]
        out=failure_summary(pd.DataFrame(rows)).set_index('category')
        self.assertEqual(out.loc[FAILURE_LABELS[0],'mean'], .375)
        self.assertEqual(out.loc[FAILURE_LABELS[3],'mean'], .625)
        self.assertTrue((out.n_questions==2).all())


if __name__ == '__main__':
    unittest.main()
