"""Replace only user-ICL prompts cut at ordinal periods by the legacy helper."""
from copy import deepcopy
import json
from control_expansion import ROOT,PRIMARY,base,cells,sentences,suite,icl,encoded


def prepare():
    donors={t.question:d for t,d in cells() if t.ctype=='numbered_sentences'}
    destination=ROOT/'numbered_icl_repair'
    for name in PRIMARY:
        old=json.loads((ROOT/name/'manifest.json').read_text());m=deepcopy(old)
        tok=suite.tokenizer(m['model']);requests=[]
        for r in m['requests']:
            if r['behavior']!='numbered_sentences' or r['condition']!='user_icl':continue
            donor=donors[r['question']]
            malformed=icl.slice_by_len(donor.controlled_cot,icl.default_spec('icl_user_plain'))
            corrected=' '.join(s for a,b,s in sentences(donor.controlled_cot)[:5])
            assert len(sentences(corrected))==min(5,len(sentences(donor.controlled_cot)))
            assert malformed!=corrected
            old_text='Internal thinking: "'+malformed+'"'
            new_text='Internal thinking: "'+corrected+'"'
            assert r['messages'][0]['content'].count(old_text)==1
            r['messages'][0]['content']=r['messages'][0]['content'].replace(old_text,new_text)
            r['prompt_token_ids']=encoded(m['model'],tok,r['messages'])
            requests.append(r)
        m['requests']=requests
        m['supersedes_manifest_sha256']=old['manifest_sha256']
        m['repair_reason']='User ICL contains five content sentences, excluding ordinal-header periods from sentence splitting.'
        m.pop('manifest_sha256');m['manifest_sha256']=base.digest(m)
        base.immutable_json(destination/name/'manifest.json',m)
        print(name,len(requests),flush=True)


if __name__=='__main__':prepare()
