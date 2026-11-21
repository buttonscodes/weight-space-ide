"""Fixed-suite semantic acceptance, independent of structural export success.

No inference here. The operator collects baseline/overlay and candidate logits in
separate guarded processes. Criteria are serialized before candidates are built.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch


@dataclass(frozen=True)
class Criteria:
    retention: float = .90
    minimum_gain: float = .1
    max_preservation_kl: float = .01
    max_logit_error: float = .05
    zero_gain_tolerance: float = .01
    direction: int = 1

    def __post_init__(self):
        for k,v in asdict(self).items():
            if not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError(f'Invalid {k}')
        if not 0 < self.retention <= 1 or self.direction not in (-1,1):raise ValueError('Invalid retention/direction')
        if min(self.minimum_gain,self.max_preservation_kl,self.max_logit_error,self.zero_gain_tolerance)<0:
            raise ValueError('Negative tolerance')

    def record(self):
        d=asdict(self)
        return dict(criteria=d,criteria_sha256=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest())


def evaluate(base, overlay, candidate, target, contrast, *, criteria, chat_prompt_identical):
    if not len(base)==len(overlay)==len(candidate) or len(base)<2:raise ValueError('Need target and unrelated fixed-suite logits')
    if any(a.shape!=base[0].shape or a.ndim!=1 or not torch.isfinite(a).all() for a in [*base,*overlay,*candidate]):
        raise ValueError('Invalid/nonfinite logits')
    margin=lambda x:float(x[target]-x[contrast])
    bm,om,cm=map(margin,(base[0],overlay[0],candidate[0]))
    gain=criteria.direction*(om-bm);actual=criteria.direction*(cm-bm)
    if gain < -1e-6:raise ValueError('Overlay contradicts declared edit direction; choose a fixed appropriate metric before candidates')
    zero=abs(gain)<=1e-6
    required=max(criteria.minimum_gain,criteria.retention*gain) if not zero else None
    gain_ok=abs(cm-om)<=criteria.zero_gain_tolerance if zero else actual>=required
    kl=lambda a,b:max(0.,float(torch.nn.functional.kl_div(a.double().log_softmax(-1),b.double().log_softmax(-1),log_target=True,reduction='sum')))
    drift=[kl(a,b) for a,b in zip(candidate[1:],base[1:])]
    top=[int(a.argmax())==int(b.argmax()) for a,b in zip(candidate[1:],base[1:])]
    error=max(float((a-b).abs().max()) for a,b in zip(candidate,overlay))
    checks=dict(gain_preserved=bool(gain_ok),unrelated_top1_preserved=all(top),
                preservation_drift=max(drift)<=criteria.max_preservation_kl,
                max_logit_error=error<=criteria.max_logit_error,chat_prompt_identical=chat_prompt_identical is True)
    return dict(validation_status='VERIFIED' if all(checks.values()) else 'FAILED',checks=checks,
                **criteria.record(),base_margin=bm,overlay_margin=om,candidate_margin=cm,
                overlay_gain=gain,candidate_gain=actual,required_gain=required,
                retention_ratio=None if zero else actual/gain,zero_gain_mode=zero,
                preservation_kl=drift,unrelated_top1=top,max_logit_error=error,
                qualification='Fixed-suite evidence only; no bit-exact or universal behavior guarantee')


def select_verified(reports):
    """First passing promotion in the predeclared sequence, never a fallback."""
    for report in reports:
        if report.get('validation_status')=='VERIFIED':return report
    raise ValueError('No candidate preserved the edit; retain the exact companion')
