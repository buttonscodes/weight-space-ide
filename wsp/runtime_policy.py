"""Interactive operation admission and escalation. No processes or tensors here.

The strict legacy acceptance Policy is deliberately unchanged.
"""
from dataclasses import asdict
from .memory_guard import GiB
from .runtime_events import event, RuntimeRefusal

ABORT_GRACE_SECONDS = 5
UNLOAD_GRACE_SECONDS = 10
GPU_STALE_GRACE_SECONDS = 5
IDLE_UNSAFE_PERSIST_SECONDS = 3
RUNAWAY_BYTES = 4*GiB


OPERATION_COMPONENTS = ('kv','kv_append','activations','logits','eager_attention',
                        'hybrid_workspace','retained_activations','retained_attention',
                        'recurrent','convolution')


def admission(settings, descriptor, sample, active=None, baseline=None, *, outside=0):
    """No RSS/forecast credit against availability, including file-backed pages.

    Cold load admits the whole configured-context forecast. A loaded operation
    admits only its newly allocated buffers at actual sequence length and growth
    of patch allowance, while static total caps still apply to the whole forecast.
    ``outside``: measured bytes outside the model forecast (a worker-hosted model's
    server footprint at first admission); the loaded cap is ``outside`` + forecast,
    and a cold load's running budget grows by ``outside`` within the static caps.
    """
    from .runtime_supervisor import policy_for
    from dataclasses import replace
    values=settings['values']
    policy=policy_for(values,sample['total'],sample['available'])
    required=max(descriptor['forecast']['total'],int((values['reservation_gib'] or 0)*GiB))
    static_cap=min(int(values[k]*GiB) for k in ('budget_gib','ceiling_gib') if values[k] is not None) if any(values[k] is not None for k in ('budget_gib','ceiling_gib')) else sample['total']-policy.preferred
    static_cap=min(static_cap,sample['total']-policy.preferred)
    loaded=bool(active and active.get('runner'))
    working=descriptor.get('working',{})
    patch_growth=max(0,descriptor['forecast'].get('patches',0)-(active or {}).get('forecast',{}).get('patches',0))
    new_bytes=sum(working.get(k,0) for k in OPERATION_COMPONENTS)+patch_growth if loaded else required
    if sample.get('pressure') != 1:
        raise RuntimeRefusal(event('memory_pressure','refused','Memory pressure is not normal; operation not started.',sample.get('pressure'),1,
            ['Free memory on the machine and retry.']))
    if measured_total(sample)>static_cap or required>static_cap or (not loaded and required>policy.budget) or sample['available']-new_bytes<policy.preferred:
        raise RuntimeRefusal(event('forecast_exceeds_budget','refused','The operation forecast exceeds current memory admission bounds.',
            dict(required=required,new_bytes=new_bytes,available=sample['available']),dict(total_cap=static_cap,preferred=policy.preferred),
            ['Free memory and retry without restarting.', 'Lower context or shorten the prompt.', 'Reduce the edit or choose a smaller model.']))
    if loaded:
        # Loaded mapped pages may have been evicted: allow them to refault on
        # top of retained private allocations, without crediting availability.
        process_cap=min(static_cap,max(outside+required,measured_total(sample)+new_bytes+descriptor['forecast'].get('weights',0)))
        policy=replace(policy,budget=process_cap,ceiling=static_cap)
    elif outside:
        # A worker-hosted cold load: available already excludes the resident server, the measured
        # total does not. Only the running limit moves (checked above unmodified); static caps bound it.
        policy=replace(policy,budget=min(static_cap,policy.budget+outside))
    # Admitted footprint ceiling of this operation: the pre-model tree plus the whole
    # forecast (a cold load faults all of it in) plus a loaded operation's new buffers.
    # Fast growth below it is admitted allocation, not runaway.
    envelope=(baseline or sample).get('tree_footprint',0)+required+(new_bytes-patch_growth if loaded else 0)
    return policy,dict(required=required,new_bytes=new_bytes,available=sample['available'],policy=asdict(policy),loaded=loaded,static_cap=static_cap,envelope=envelope,outside=outside)



def largest_patch(settings, descriptor, sample, active=None, baseline=None, *, outside=0, per_weight=12):
    """Advisory and pure: the largest number of new patch weights ``admission`` admits now.

    ``descriptor`` is the edit descriptor ``preflight_edit`` would send with no new patch (its working
    scratch included); each patch weight adds ``per_weight`` bytes to its forecast, as ``_value`` does.
    Every admission bound grows with the patch, so a bisection finds the edge exactly. Returns
    ``(largest, refusal)``: the refusal ``admission`` gives one weight past it (``None`` when unbounded).
    Nothing is bound, reserved, published or logged. The answer is exact for a fresh admission of the edit.
    While another operation runs it can be more permissive than the later real grant: patch growth is counted
    against the active descriptor's patches and clamps at zero while those exceed the new descriptor's.
    """
    def trial(n):
        value=dict(descriptor);value['forecast']=dict(descriptor['forecast'])
        value['forecast']['patches']=value['forecast'].get('patches',0)+n*per_weight
        value['forecast']['total']=value['forecast']['total']+n*per_weight
        try:admission(settings,value,sample,active,baseline,outside=outside)
        except RuntimeRefusal as exc:return exc.record
        return None
    first=trial(0)
    if first is not None:return 0,first
    high=max(1,int(sample['total'])//per_weight+1)
    if trial(high) is None:return high,None
    low=0
    while high-low>1:
        mid=(low+high)//2
        if trial(mid) is None:low=mid
        else:high=mid
    return low,trial(low+1)


def extend_admission(policy, decision, current, *, weights=0, edit=False):
    """An explicit, freshly admitted extension may grow the otherwise fixed cap."""
    from dataclasses import replace
    projected=(measured_total(current)+decision['new_bytes']+weights if decision['loaded']
               else max(decision['required'],measured_total(current)))
    if (decision['required']>decision['static_cap'] or measured_total(current)>decision['static_cap']
        or decision['new_bytes']>max(0,current['available']-policy.preferred)):
        raise RuntimeRefusal(event('edit_exceeds_budget' if edit else 'operation_exceeds_budget','refused',
            'The edit exceeds current admission bounds.' if edit else 'The operation extension exceeds current admission bounds.',
            dict(required=decision['required'],projected=projected,new_bytes=decision['new_bytes']),decision['static_cap'],
            ['Reduce the edit and retry.'] if edit else ['Shorten the operation or free memory and retry.']))
    limit=min(decision['static_cap'],max(policy.budget,projected))
    return replace(policy,budget=limit,ceiling=decision['static_cap'])


def measured_total(sample):
    gpu=sample.get('gpu') or {}
    return max(sample.get('tree_footprint',0),sample.get('tree_rss',0),gpu.get('metal_allocated',0) or 0,gpu.get('mps_driver',0) or 0)


def observations(policy,sample,baseline,previous=None,envelope=None):
    """First unsafe condition plus warnings; swap/compression alone never abort.

    With a running operation's admitted envelope, fast growth is runaway only once
    the tree footprint is above that envelope (the budget check still precedes it).
    """
    baseline=baseline or sample
    warnings=[]
    for code,field,limit,label in [('swap_growth','swap_used',policy.swap_growth,'Swap'),
                                  ('compression_growth','compressed',policy.compression_growth,'Compression')]:
        growth=max(0,sample.get(field,0)-baseline.get(field,0))
        if code=='swap_growth':growth=max(growth,sample.get('swap_out',0)-baseline.get('swap_out',0))
        if growth>limit:
            warnings.append(event(code,'warning',f'{label} growth exceeds the configured warning tolerance.',growth,limit,
                ['Free memory if responsiveness degrades.', f'Review the {label.lower()} tolerance in Settings.']))
    if sample.get('pressure')!=1:
        return event('memory_pressure','abort','Memory pressure is not normal; stopping this operation.',sample.get('pressure'),1,
            ['Free memory on the machine and retry.']),warnings
    if sample.get('available',0)<policy.floor:
        return event('headroom_floor','abort','Available memory is below the runtime floor.',sample.get('available'),policy.floor,
            ['Free memory or unload the model.']),warnings
    footprint=measured_total(sample)
    if footprint>min(policy.budget,policy.ceiling):
        return event('runtime_budget','abort','Measured runtime memory exceeds the allowed total.',footprint,min(policy.budget,policy.ceiling),
            ['Reduce context or edits.', 'Unload the model if memory does not recover.']),warnings
    if (previous and sample.get('tree_footprint',0)-previous.get('tree_footprint',0)>RUNAWAY_BYTES
            and (envelope is None or sample.get('tree_footprint',0)>envelope)):
        return event('runaway_growth','abort','Runtime memory grew by more than 4 GiB in one sample'+
            (' and exceeds the admitted operation envelope.' if envelope is not None else '.'),sample['tree_footprint']-previous['tree_footprint'],RUNAWAY_BYTES,
            ['Unload the model and inspect the operation before retrying.']),warnings
    return None,warnings


class Escalation:
    """Bounded cooperative abort (step 2) and unload (step 3), then ``kill_worker`` (step 4).

    Step 4 stops only the model worker process; the supervisor never signals the
    server for a memory reason. With no live worker step 4 is a recorded no-op.
    """
    def __init__(self):
        self.action=None;self.since=None;self.record=None;self.gpu_stale_since=None;self.idle_unsafe_since=None;self.idle_clear_since=None

    def reset(self):
        self.action=self.since=self.record=self.gpu_stale_since=None;self.idle_unsafe_since=None;self.idle_clear_since=None

    def acknowledge_finish(self, healthy, now):
        """An acknowledged timely abort may keep a freshly proven healthy model."""
        if (self.action=='abort' and healthy and now-self.since<ABORT_GRACE_SECONDS
            and self.record['code'] not in ('model_changed','runaway_growth')):
            self.reset()
            return True
        return False

    def observe(self,issue,running,now,*,idle_expired=False,gpu_stale=False):
        # Once revoked, recovery of the original symptom never renews the lease.
        # Only an acknowledged release resets this ladder.
        if self.action == 'unload':
            if now-self.since >= UNLOAD_GRACE_SECONDS:
                self.action='kill_worker'
            return self.action
        if self.action == 'abort':
            if (self.record['code']=='runaway_growth' and issue and
                issue['code']=='runaway_growth' and now>self.since):
                self.action='kill_worker'
            elif not running or now-self.since >= ABORT_GRACE_SECONDS:
                self.action='unload';self.since=now
            return self.action
        if self.action == 'kill_worker':return self.action
        gpu_stale=bool(gpu_stale and issue and issue['code']=='gpu_telemetry_stale')
        if gpu_stale:
            if self.gpu_stale_since is None:self.gpu_stale_since=now
            if not running or now-self.gpu_stale_since<GPU_STALE_GRACE_SECONDS:
                return 'blocked'
        else:self.gpu_stale_since=None
        if not running and issue and issue['code'] in ('memory_pressure','headroom_floor') and not idle_expired:
            if self.idle_clear_since is not None and now-self.idle_clear_since>=IDLE_UNSAFE_PERSIST_SECONDS:
                self.idle_unsafe_since=None
            self.idle_clear_since=None
            if self.idle_unsafe_since is None:self.idle_unsafe_since=now
            if now-self.idle_unsafe_since<IDLE_UNSAFE_PERSIST_SECONDS:return 'blocked'
        elif running or issue is not None or idle_expired:self.idle_unsafe_since=self.idle_clear_since=None
        elif self.idle_unsafe_since is not None:
            # A condition that flaps must not hold an idle model forever: the
            # debounce restarts only once the condition has stayed clear as long.
            if self.idle_clear_since is None:self.idle_clear_since=now
            if now-self.idle_clear_since>=IDLE_UNSAFE_PERSIST_SECONDS:self.idle_unsafe_since=self.idle_clear_since=None
        if issue is None and not idle_expired:return None
        self.action='abort' if running else 'unload'
        self.since=now;self.record=issue
        return self.action
