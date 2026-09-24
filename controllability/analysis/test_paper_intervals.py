import unittest
import pandas as pd
from paper_intervals import question_cluster_ci


class ClusterIntervalsTest(unittest.TestCase):
    def test_shared_questions_across_behaviors(self):
        # Opposite behavior scores cancel for every question. Independently
        # resampling behavior rows would introduce spurious uncertainty.
        d = pd.DataFrame([dict(question=str(q), ctype=b, compliance_score=v)
                          for q in range(10) for b,v in [('a',q%2),('b',1-q%2)]])
        r = question_cluster_ci(d, 'question', ['a','b'], n_boot=1000)
        self.assertEqual((r['mean_compliance'],r['ci_low'],r['ci_high']),(.5,.5,.5))

    def test_duplicating_samples_does_not_shrink_intervals(self):
        d = pd.DataFrame(dict(question=[str(i) for i in range(10)],
                             ctype=['a']*10, compliance_score=[0]*5+[1]*5))
        a = question_cluster_ci(d,'question',['a'],n_boot=1000)
        b = question_cluster_ci(pd.concat([d]*5),'question',['a'],n_boot=1000)
        for key in ['mean_compliance','ci_low','ci_high']:
            self.assertAlmostEqual(a[key],b[key])

    def test_unequal_samples_preserve_existing_point_estimate(self):
        d = pd.DataFrame(dict(question=['a','a','a','b'],ctype=['x']*4,
                             compliance_score=[0,0,0,1]))
        r = question_cluster_ci(d,'question',['x'],n_boot=1000)
        self.assertEqual(r['mean_compliance'],.25)

    def test_missing_behavior_is_rejected(self):
        d = pd.DataFrame(dict(question=['a'],ctype=['x'],compliance_score=[1]))
        with self.assertRaises(ValueError):
            question_cluster_ci(d,'question',['x','y'],n_boot=100)


if __name__ == '__main__':
    unittest.main()
