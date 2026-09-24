import json
import subprocess
import unittest
import numpy as np
import pandas as pd
from monitorability.control_type_viewer import ROOT, OUT, FAMILIES, correlation


class CorrelationTests(unittest.TestCase):
    def test_known_signs(self):
        for y, expected in [([2, 4, 6], 1), ([6, 4, 2], -1)]:
            got = correlation([1, 2, 3], y)
            self.assertAlmostEqual(got['pearson'], expected)
            self.assertAlmostEqual(got['spearman'], expected)

    def test_ties_have_average_ranks(self):
        self.assertAlmostEqual(correlation([1, 1, 2], [1, 2, 3])['spearman'], np.sqrt(.75))

    def test_missing_constant_and_small_n(self):
        self.assertEqual(correlation([1, np.nan, 2], [3, 4, 5])['n'], 2)
        self.assertIsNone(correlation([1, 1, 1], [1, 2, 3])['pearson'])
        self.assertIsNone(correlation([1, 2, 3], [0, 0, 0])['spearman'])

    def test_family_weights_and_sources(self):
        data = json.loads((OUT/'data.json').read_text())
        self.assertEqual(len(data['models']), 12)
        self.assertEqual(FAMILIES['suppression'], ['omit_word', 'no_first_person'])
        for model in data['models']:
            for family, members in FAMILIES.items():
                for metric in ['soft', 'hard']:
                    self.assertAlmostEqual(model['control']['family:'+family][metric],
                        np.mean([model['control'][c][metric] for c in members]))
        self.assertEqual(len(data['counts']), 96)
        self.assertTrue(all(c['questions']==30 for c in data['counts']))

    def test_javascript_matches_python_for_all_views(self):
        code = """
const fs=require('fs'),{stats}=require('./monitorability/control_type_viewer.js');
const d=JSON.parse(fs.readFileSync('results/monitorability/control_type_viewer/data.json'));
const rows=[];
for(const metric of ['soft','hard'])for(const control of [...d.types,...Object.keys(d.families).map(f=>'family:'+f)])for(const ev of d.evals){
 rows.push({control,compliance_metric:metric,evaluation:ev.key,...stats(d.models.map(m=>({x:m.control[control][metric],y:m.monitor[ev.key]})))});
}
rows.push({edge:true,...stats([{x:1,y:3},{x:1,y:2},{x:2,y:1},{x:null,y:0}])});
console.log(JSON.stringify(rows));
"""
        js=json.loads(subprocess.check_output(['node','-e',code],cwd=ROOT,text=True))
        expected=pd.read_csv(OUT/'correlations.csv').set_index(['control','compliance_metric','evaluation'])
        self.assertEqual(len(js)-1, 220)
        for row in js[:-1]:
            e=expected.loc[(row['control'],row['compliance_metric'],row['evaluation'])]
            self.assertEqual(row['n'],e['n'])
            for metric in ['pearson','spearman']:
                if pd.isna(e[metric]): self.assertIsNone(row[metric])
                else:self.assertAlmostEqual(row[metric],e[metric],places=12)
        self.assertEqual(js[-1]['n'],3)
        self.assertAlmostEqual(js[-1]['spearman'],-np.sqrt(.75))

if __name__=='__main__':unittest.main()
