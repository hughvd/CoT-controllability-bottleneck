import unittest
from controllability.control_expansion import score_text, sentences

class BoundaryTests(unittest.TestCase):
    def test_number_headers_are_not_sentences(self):
        self.assertEqual(len(sentences('1. First. 2. Second.')),2)
        self.assertEqual(score_text('numbered_sentences','','1. First. 2. Second.')[0],1)
        self.assertEqual(score_text('numbered_sentences','','1. First. 3. Second.')[0],.5)
    def test_number_prefill_retains_ordinal(self):
        self.assertEqual(score_text('numbered_sentences','1. Fir','st. 2. Second.')[0],1)
        self.assertEqual(score_text('numbered_sentences','1. Fir','st. 1. Second.')[0],0)
        self.assertIsNone(score_text('numbered_sentences','1. Fir','st.')[0])
    def test_suffix_boundary(self):
        self.assertEqual(score_text('suffix_sentences','First PI','NG. Next PING.','PING')[0],1)
        self.assertEqual(score_text('suffix_sentences','First PING. ','Next fails.','PING')[0],0)
        self.assertIsNone(score_text('suffix_sentences','First PING','.','PING')[0])
    def test_empty_is_not_success(self):
        self.assertEqual(score_text('suffix_sentences','First PING.','','PING')[0],0)
    def test_suffix_after_punctuation_is_same_sentence(self):
        for text in ['First PING. Second PING.', 'First. PING\nSecond. PING',
                     'First. PING. Second. PING.', 'First! PING\nSecond? PING']:
            self.assertEqual(score_text('suffix_sentences','',text,'PING'),(1.,2))
        self.assertEqual(score_text('suffix_sentences','','First. PING\nSecond fails.','PING'),(.5,2))
        self.assertEqual(score_text('suffix_sentences','','1. First. PING\n2. Second PING.','PING'),(1.,2))
    def test_suffix_post_punctuation_prefill_boundary(self):
        self.assertEqual(score_text('suffix_sentences','First. PI','NG\nSecond. PING','PING'),(1.,2))
        self.assertEqual(score_text('suffix_sentences','First.',' PING','PING'),(1.,1))
        self.assertEqual(score_text('suffix_sentences','First. PING\n','Second fails.','PING'),(0.,1))
        self.assertIsNone(score_text('suffix_sentences','First. PING','.','PING')[0])
    def test_suffix_requires_terminal_standalone_marker(self):
        for text in ["I will append 'PING' to every sentence.", 'First NOTPING.', 'PING.']:
            self.assertEqual(score_text('suffix_sentences','',text,'PING')[0],0)
    def test_numbering_preserves_existing_partial_credit(self):
        self.assertAlmostEqual(score_text('numbered_sentences','','1. First. Extra sentence. 2. Next.')[0],2/3)
        self.assertAlmostEqual(score_text('numbered_sentences','1. Fir','st. Extra sentence. 2. Next.')[0],.5)

if __name__=='__main__':unittest.main()
