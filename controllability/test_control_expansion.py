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

if __name__=='__main__':unittest.main()
