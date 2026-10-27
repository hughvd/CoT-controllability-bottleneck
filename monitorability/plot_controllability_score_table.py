"""Appendix controllability scores, with question-cluster bootstrap uncertainty.

Run: python -m monitorability.plot_controllability_score_table
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from monitorability.plot_monitorability_score_table import draw_score_table
from monitorability.plot_control_monitor_heatmap import CONTROLS, OUT, SOURCE
from monitorability.sweep_analysis import ROOT, controllability_frame

NAME = 'controllability_score_uncertainty_table'
SEED = 20260925
NBOOT = 5000
CAPTION = (
    'Controllability scores and uncertainty across the twelve-model suite. '
    'Entries show mean soft compliance with instructions and no prefill, '
    'using the same scores as the correlation analysis, ± one bootstrap standard '
    'deviation from 5,000 question-level resamples. All valid responses to each '
    'sampled question are retained together. Columns separate information addition, '
    'stylistic modification, and information suppression. The bottom row reports '
    'the minimum and maximum model scores for each task. Light blue shading '
    'encodes score magnitude on a shared 0–1 scale. These are uncertainties in '
    'mean compliance, not confidence intervals for the correlations.'
)


def main():
    data = json.loads(SOURCE.read_text())
    sources = {str(SOURCE.relative_to(ROOT)): hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    from controllability.rescore_first_person import provenance
    sources.update(provenance())
    rows = []
    for model in data['models']:
        old_path = ROOT / 'results/prefill_continuation' / model['run'] / 'sweep.json'
        new_path = ROOT / 'controllability/results/control_expansion/v1/correlation' / model['model'] / 'scores.csv'
        for path in [old_path,new_path]:
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        old = controllability_frame(model['run'])
        old = old[old.condition.eq('prefill_plus_instruction') & old.ctype.isin(CONTROLS)]
        extra = pd.read_csv(new_path).rename(columns={'behavior':'ctype'})
        frame = pd.concat([old,extra],ignore_index=True)
        for control in CONTROLS:
            cell = frame[frame.ctype.eq(control)]
            assert cell.compliance_score.between(0,1).all() and cell.question.notna().all()
            score = cell.compliance_score.mean()
            assert np.isclose(score,model['control'][control]['soft'],atol=1e-12,rtol=0), (model['model'],control)
            clusters = cell.groupby('question',sort=True).compliance_score.agg(['sum','count'])
            assert len(clusters)==30, (model['model'],control,len(clusters))
            # Ratio of sums preserves the response-weighted mean when a few
            # historical questions have fewer than five valid responses.
            indices = np.random.default_rng(SEED).integers(0,len(clusters),size=(NBOOT,len(clusters)))
            draws = clusters['sum'].to_numpy()[indices].sum(axis=1) / clusters['count'].to_numpy()[indices].sum(axis=1)
            rows.append(dict(model=model['model'],control=control,score=score,
                             bootstrap_sd=draws.std(ddof=1),n_questions=len(clusters),n_responses=len(cell)))
    frame = pd.DataFrame(rows)
    assert len(frame)==96 and not frame.duplicated(['model','control']).any()
    models = [m['model'] for m in data['models']]
    scores = frame.pivot(index='model',columns='control',values='score').loc[models,CONTROLS]
    deviations = frame.pivot(index='model',columns='control',values='bootstrap_sd').loc[models,CONTROLS]
    ranges = frame.groupby('control').score.agg(['min','max']).loc[CONTROLS]
    ranges['span'] = ranges['max']-ranges['min']
    frame.to_csv(OUT/f'{NAME}.csv',index=False)
    ranges.to_csv(OUT/f'{NAME}_ranges.csv')
    draw_score_table(scores,deviations,ranges,NAME,
                     ['Sentence\nprefixes','Numbered\nsentences','Sentence\nendings','Uppercase','Leetspeak','Spanish','Omit\nword','No first\nperson'],
                     [('Addition',1),('Style',4),('Suppression',6.5)],[2.5,5.5])
    (OUT/f'{NAME}_caption.md').write_text(CAPTION+'\n')
    (OUT/f'{NAME}.tex').write_text(
        '\\begin{figure}[t]\n    \\centering\n'
        f'    \\includegraphics[width=\\linewidth]{{figures/{NAME}.pdf}}\n'
        '    \\caption{'+CAPTION.replace('±','$\\pm$')+'}\n'
        '    \\label{fig:controllability-score-uncertainty}\n\\end{figure}\n')
    provenance = dict(sources=sources,bootstrap_replicates=NBOOT,seed=SEED,
                      resampling_unit='question; all valid responses retained as a cluster',
                      estimator='response-weighted mean compliance',uncertainty='bootstrap standard deviation (ddof=1)',
                      color_scale=[0,1],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/f'{NAME}_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(ranges.round(3).to_string())


if __name__ == '__main__':
    main()
