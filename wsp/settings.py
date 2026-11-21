"""Persistent runtime settings and advisory previews. No model loads or tensor reads."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading

from .runtime_config import MEMORY, NEXT_LOAD, SERVER_LIVE, SUPERVISOR_LIVE, _validate, backend_status, resolve, runtime_thread_limit
from .capacity import MAX_EXACT_INTEGER, DEFAULT_CONTEXT, DEFAULT_OUTPUT, PREFILL_CHUNK, NATIVE_ROWS, context_limit

EDITABLE = {'backend', 'threads', 'context', 'new_tokens', 'resident_seconds'} | MEMORY
GiB = 1024**3
RECOMMENDATION_BUFFER_MIN = 512 * 1024**2
RECOMMENDATION_BUFFER_FRACTION = 0.05


class SettingsConflict(ValueError):
    pass


class SettingsOutOfStep(RuntimeError):
    """The settings file was not written, and the supervisor runs other settings or it is not known which it runs
    (``SettingsStore.out_of_step`` says which until the next accepted save)."""


class SettingsNeedConfirmation(RuntimeError):
    """Nothing was saved: the supervisor would unload the loaded model with these settings (``effect``). Saving the
    same candidate again with ``confirm=token`` applies them."""
    def __init__(self, message, effect, token):
        super().__init__(message)
        self.effect = effect; self.token = token


def supervised(runtime):
    """Whether ``runtime`` is a supervisor client that saved settings can be applied to."""
    return runtime is not None and callable(getattr(runtime, 'update_settings', None))


def takes_effect(key, managed):
    """When a saved change of ``key`` takes effect: 'saved' (at once), 'next_load' or 'next_start'."""
    if key in SERVER_LIVE: return 'saved'
    if not managed: return 'next_start'
    return 'saved' if key in SUPERVISOR_LIVE else 'next_load'


def _same_notices(a, b):
    return [(n.get('code'), n.get('message')) for n in a or ()] == [(n.get('code'), n.get('message')) for n in b or ()]


def revision(raw):
    return hashlib.sha256(raw if raw is not None else b'<missing>').hexdigest()


class SettingsStore:
    def __init__(self, active):
        self.active = copy.deepcopy(active)
        self.path = Path(active['config'])
        self.profile = active.get('profile')
        self.lock = threading.RLock()
        # Set when the supervisor runs settings other than the file's, or it is not known which: the page says so,
        # and the next save is sent to the supervisor even when it changes nothing (an accepted save clears it).
        self.out_of_step = None
        # (document, revision, active): one consistent state, kept as the lock is released, for status reads that
        # must not wait behind a save in flight (consistent()).
        self._view = None
        # activate() rewrites the child's environment. Never re-resolve from that environment.
        self.overrides = {k: active['values'][k] for k, source in active['sources'].items()
                          if source == 'command line' or source.startswith('environment ')}
        try: self._view = (*self.read(), self.active)
        except (ValueError, OSError): pass   # read again (and reported) when it is asked for

    def read(self):
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise ValueError('Settings editor requires a regular configuration file, not a symlink or special file')
        if self.path.exists():
            if self.path.stat().st_size > 1024*1024:
                raise ValueError('Configuration exceeds the 1 MiB settings editor limit')
            raw = self.path.read_bytes()
            try: document = json.loads(raw)
            except ValueError as exc: raise ValueError('Configuration is not valid JSON; repair it before saving') from exc
        else:
            raw, document = None, {'version': 1, 'defaults': {}, 'profiles': {}}
            if self.profile:
                raise ValueError('The active profile configuration is missing; restore it before saving')
        self.resolve(document)  # validate without mutating any file
        return document, revision(raw)

    def resolve(self, document, effective=False):
        saved = resolve(str(self.path), self.profile, env={}, system=self.active['platform'], document=document)
        if effective:
            saved['values'].update(self.overrides)
            saved['sources'].update({k: self.active['sources'][k] for k in self.overrides})
            # A launch thread override replaces the clamped saved value at the next start.
            if 'threads' in self.overrides:
                saved['notices'] = [n for n in saved['notices'] if n['code'] != 'threads_clamped']
        return saved

    def candidate(self, document, changes):
        if not isinstance(changes, dict) or changes.keys() - EDITABLE:
            raise ValueError('Only runtime and memory settings may be changed here')
        _validate(changes, 'settings', self.active.get('thread_limit') or runtime_thread_limit(self.active['platform']))
        result = copy.deepcopy(document)
        target = result.setdefault('defaults', {}) if not self.profile else result['profiles'][self.profile]
        target.update(changes)
        self.resolve(result)
        return result

    def updated(self, effective, managed):
        """The active settings after saving ``effective`` (saved values with launch overrides still winning):
        every setting the running server applies now or at the next model load takes the new value."""
        result = copy.deepcopy(self.active)
        for key in SERVER_LIVE | ((SUPERVISOR_LIVE | NEXT_LOAD) if managed else set()):
            result['values'][key] = effective['values'][key]
            result['sources'][key] = effective['sources'][key]
        if managed and not _same_notices(result.get('notices'), effective['notices']):
            result['notices'] = copy.deepcopy(effective['notices'])
        return result

    def describe(self, document, rev, runtime=None, active=None):
        saved = self.resolve(document)
        effective = self.resolve(document, effective=True)
        active = active if active is not None else self.active
        managed = supervised(runtime)
        changed = [k for k in EDITABLE if effective['values'][k] != active['values'][k]]
        overridden = [k for k in EDITABLE if k in self.overrides and saved['values'][k] != effective['values'][k]]
        # A saved value that differs from the running one was written outside this page (or before a start
        # without a supervisor): managed servers apply it at the next save, otherwise it waits for a start.
        pending = {k: 'next_save' if managed and takes_effect(k, True) != 'next_start' else 'next_start' for k in changed}
        when = {k: 'override' if k in self.overrides else takes_effect(k, managed) for k in EDITABLE}
        loaded = runtime.loaded_config() if managed else None
        if managed:
            note = ('Saving applies settings to this server: memory limits, the idle timeout, default output tokens and the largest weight patch '
                    'at once (a running operation keeps the memory limits it was admitted with until it finishes); context, CPU threads and backend '
                    'at the next model load. Command-line and original environment overrides still win.')
        else:
            note = ('This server has no memory supervisor. Default output tokens and the largest weight patch apply as soon as they are saved; '
                    'every other setting is saved for the next start of a supervised server (regular wsp serve). Command-line and original '
                    'environment overrides still win when reused.')
        return dict(thread_limit=active.get('thread_limit') or runtime_thread_limit(active['platform']), revision=rev, config=str(self.path), profile=self.profile, target='profile '+self.profile if self.profile else 'config defaults',
                    active={k: active[k] for k in ('values','sources','platform')},
                    saved={k: saved[k] for k in ('values','sources')},
                    next_start={k: effective[k] for k in ('values','sources')},
                    changed=sorted(changed), overridden=sorted(overridden), pending=pending,
                    restart_required=any(v == 'next_start' for v in pending.values()),
                    takes_effect=when, supervised=managed, loaded=loaded,
                    live=sorted(k for k in EDITABLE if takes_effect(k, managed) == 'saved'),
                    next_load=sorted(NEXT_LOAD) if managed else [],
                    out_of_step=self.out_of_step if managed else None,
                    note=note)

    def consistent(self):
        """-> (document, revision, active) from one moment. Without waiting for a save in flight, which holds the
        lock across the supervisor exchange (seconds): the state as that save found it, read together under the
        lock, never the new file with the old active settings."""
        if self.lock.acquire(blocking=False):
            try:
                document, rev = self.read()
                self._view = (document, rev, self.active)
                return self._view
            finally: self.lock.release()
        view = self._view
        if view is not None: return view
        with self.lock:
            document, rev = self.read()
            self._view = (document, rev, self.active)
            return self._view

    def snapshot(self, runtime=None):
        document, rev, active = self.consistent()
        return self.describe(document, rev, runtime, active)

    @staticmethod
    def confirmation(expected, updated, judged):
        """The token a confirmed save of exactly this candidate carries: bound to the file revision it starts from,
        to the settings it would apply and to the effect the supervisor judged (``judged``, its fingerprint of the
        unload and of the model and operation it concerns), so a confirmation of one draft, or of one model state,
        cannot authorise another. -> '<binding>.<judged>'."""
        body = json.dumps([expected, updated['values'], updated['sources'], judged], sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()+'.'+str(judged)

    def _confirmed(self, confirm, expected, updated):
        # The effect fingerprint a valid token for this candidate carries, else None (unconfirmed).
        if not isinstance(confirm, str) or '.' not in confirm: return None
        judged = confirm.rsplit('.', 1)[1]
        return judged if judged and confirm == self.confirmation(expected, updated, judged) else None

    def save(self, expected, changes, validate=None, runtime=None, confirm=None):
        """Write the settings file and apply the change to the running server (see takes_effect).

        With a supervisor (``supervised(runtime)``) the supervisor validates and applies the new settings first.
        Its refusal, or no answer while it still holds the previous settings, raises with nothing written or
        changed; a change that would unload the loaded model raises SettingsNeedConfirmation unless ``confirm`` is
        that candidate's token. The file is written only after it accepted; if the write fails the supervisor is
        returned to the previous settings and the failure raised. SettingsOutOfStep: the supervisor runs other
        settings, or it is not known which (``out_of_step``); ``active`` then follows the supervisor client.
        -> describe() plus ``applied``: what happened to each change."""
        # Advisory cross-process lock plus revision conflict protects concurrent IDE saves.
        # An external editor that ignores this lock must still be reconciled on the next save.
        import fcntl
        from .runtime_events import RuntimeRefusal
        managed = supervised(runtime)
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(self.path)+'.lock', 'a') as lockfile:
                fcntl.flock(lockfile, fcntl.LOCK_EX)
                document, rev = self.read()
                if expected != rev: raise SettingsConflict('Settings changed on disk. Reload Settings before saving again.')
                candidate = self.candidate(document, changes)
                if validate:
                    validate(self.resolve(candidate))
                    validate(self.resolve(candidate, effective=True))
                effective = self.resolve(candidate, effective=True)
                previous = copy.deepcopy(self.active)
                updated = self.updated(effective, managed)
                saved_before, saved_after = self.resolve(document)['values'], self.resolve(candidate)['values']
                raw = (json.dumps(candidate, indent=2, ensure_ascii=False)+'\n').encode()
                answer = None
                if managed and (updated != previous or self.out_of_step):
                    try:
                        answer = runtime.update_settings(updated, confirmed=self._confirmed(confirm, expected, updated))
                    except RuntimeRefusal as exc:
                        if exc.record.get('code') == 'settings_unconfirmed':
                            effect = getattr(exc, 'effect', None)
                            token = self.confirmation(expected, updated, (effect or {}).get('judged'))
                            raise SettingsNeedConfirmation('Not saved: '+str(exc), effect, token) from exc
                        if hasattr(exc, 'held'):
                            # Not known, or other settings, in the supervisor: nothing written; say so.
                            self._out_of_step(runtime, document, rev, f'Not saved: {exc} The settings file was not changed.')
                            raise SettingsOutOfStep(self.out_of_step) from exc
                        raise    # refused, or not applied: nothing written or changed
                try:
                    unsynced = self._write(raw, expected)
                except BaseException as exc:
                    if answer is not None:
                        try: runtime.update_settings(previous, restore=True)
                        except Exception as again:
                            if hasattr(again, 'held'):
                                text = f'Not saved: {exc}. Returning the memory watchdog to the previous settings: {again}'
                            else:
                                # Refused or not applied: the supervisor runs the unsaved values. The page must say so.
                                text = (f'Not saved: {exc}. The memory watchdog could not be returned to the previous settings ({again}); '
                                        'it uses the unsaved values until the next accepted save or a server restart.')
                            self._out_of_step(runtime, document, rev, text, fallback=updated)
                            raise SettingsOutOfStep(text) from exc
                    raise
                self.active = updated; self.out_of_step = None
                self._view = (candidate, revision(raw), self.active)
                result = self.describe(candidate, revision(raw), runtime)
                keys = {k for k in EDITABLE if updated['values'][k] != previous['values'][k] or saved_after[k] != saved_before[k]}
                applied = dict(now=[], next_load=[], next_start=[], overridden=[])
                for key in sorted(keys):
                    when = 'overridden' if key in self.overrides else {'saved': 'now'}.get(takes_effect(key, managed), takes_effect(key, managed))
                    applied[when].append(key)
                answer = answer or {}
                result['applied'] = dict(applied, supervised=managed, running=bool(answer.get('running')), effect=answer.get('effect'),
                                         late=bool(answer.get('late')), loaded=runtime.loaded_config() if managed else None, warning=unsynced)
                return result

    def _out_of_step(self, runtime, document, rev, text, fallback=None):
        # The file is unchanged. Active settings follow the supervisor client, which holds what the supervisor was
        # last known to run (or adopted what it said it runs); the page shows out_of_step until an accepted save.
        held = getattr(runtime, 'settings', None)
        self.active = copy.deepcopy(held) if isinstance(held, dict) and 'values' in held else (fallback or self.active)
        self.out_of_step = text
        self._view = (document, rev, self.active)

    def _write(self, raw, expected):
        # Atomic replace, after a recheck that nobody changed the file meanwhile. Called under both locks.
        fd, temporary = tempfile.mkstemp(prefix='.'+self.path.name+'-', suffix='.tmp', dir=self.path.parent)
        try:
            mode = (self.path.stat().st_mode & 0o777) if self.path.exists() else 0o600
            os.fchmod(fd, mode)
            with os.fdopen(fd, 'wb') as out:
                out.write(raw); out.flush(); os.fsync(out.fileno())
            # Recheck immediately before replace, including external non-cooperating edits.
            if self.read()[1] != expected: raise SettingsConflict('Settings changed while saving. Reload and retry.')
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        # The file is replaced: from here on it is saved, so a failure is reported, not undone.
        try:
            directory = os.open(self.path.parent, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
        except OSError as exc:
            return f'The settings file was replaced, but its directory could not be synced to disk: {exc}'
        return None


def validate_interactive(settings, total=None):
    values = settings['values']
    _validate(values, 'settings', settings.get('thread_limit') or runtime_thread_limit(settings.get('platform')))
    for key in ('floor_gib','preferred_gib','budget_gib','ceiling_gib','patch_limit_gib'):
        if values[key] is not None and values[key] <= 0: raise ValueError(f'{key} must be positive or Auto')
    for low, high in [('floor_gib','preferred_gib'), ('budget_gib','ceiling_gib')]:
        if values[low] is not None and values[high] is not None and not 0 < values[low] <= values[high]:
            raise ValueError(f'Memory policy requires 0 < {low} <= {high}')
    if total:
        from .runtime_supervisor import policy_for
        policy_for(values, total)


def preview(settings, ws=None, runtime=None, telemetry=None):
    """Pure metadata forecast + machine telemetry. Estimates never reserve memory or admit a load."""
    from .runtime_supervisor import policy_for
    from .memory_guard import snapshot, forecast
    values = settings['values']
    backend = backend_status(settings)
    model_context = getattr(getattr(ws, 'struct', None), 'n_ctx', None) if ws is not None else None
    context = context_limit(values['context'], model_context)
    errors, warnings, info = [], [], []
    # Notices from resolving the saved values (for example a clamped thread count).
    warnings.extend(n['message'] for n in settings.get('notices', ()))
    if model_context and values['context'] > model_context: warnings.append(f'Model context capacity is {model_context}; the runner will use the lower limit')
    if values['new_tokens'] >= context: info.append('Saved output is the default, not a maximum. Each request can ask for any output count up to the space remaining after its complete prompt and history.')
    try:
        machine = (telemetry or snapshot)()
        if not 0 < machine['available'] <= machine['total']: raise ValueError('Invalid memory telemetry')
        if machine.get('pressure') != 1: warnings.append('Memory pressure is not normal; the watchdog will refuse admission')
    except Exception as exc:
        machine = {'error': str(exc)}
        warnings.append('Memory telemetry unavailable; headroom cannot be assessed')
    policy = None
    if not machine.get('total'):
        try: validate_interactive(settings)
        except ValueError as exc: errors.append(str(exc))
    if machine.get('total'):
        try:
            from dataclasses import asdict
            policy = asdict(policy_for(values, machine['total'], machine['available']))
        except ValueError as exc: errors.append(str(exc))
    estimate = None
    if ws is not None and ws.kind == 'gguf':
        try: estimate = forecast(ws, context, operation='interactive')
        except Exception as exc: warnings.append('Model forecast unavailable: '+str(exc))
    elif ws is not None:
        warnings.append('This memory policy and forecast apply to the packed GGUF runtime; other runners are not covered')
    required = max(estimate['total'], int((values['reservation_gib'] or 0)*GiB)) if estimate else None
    active_required = None
    active_policy = None
    active_operation = False
    active_static_cap = None
    resident = bool(ws is not None and getattr(ws, 'runner_loaded', False))
    if runtime is not None and ws is not None and getattr(ws, 'runner_loaded', False):
        try:
            status = runtime.status(ws)
            reserved = bool(status.get('reserved'))
            reservation = status.get('reservation', {})
            active_policy = reservation.get('policy') if reserved else None
            active_operation = bool(reservation.get('operation_id'))
            active_static_cap = (reservation.get('admission') or {}).get('static_cap') if reserved else None
            if reserved and not active_operation and isinstance(reservation.get('watch'), dict):
                # Idle: the watch applies the saved values (runtime_supervisor.idle_watch_policy), not the last admission's.
                active_policy = reservation['watch']
                active_static_cap = min(active_policy['budget'], active_policy['ceiling'])
            # A busy status (another thread holds the runtime lock) may have no forecast yet:
            # an unknown forecast gives no resident credit, never an error.
            active_forecast = status.get('forecast')
            if reserved and isinstance(active_forecast, dict) and isinstance(active_forecast.get('total'), int):
                active_required = max(active_forecast['total'], int((runtime.settings['values']['reservation_gib'] or 0)*GiB))
            elif reserved:
                warnings.append('Active reservation forecast unknown'+(' while the runtime is busy' if status.get('busy') else '')+'; no resident credit has been assumed')
        except (KeyError, ValueError, TypeError, OSError):
            warnings.append('Active reservation unknown; no resident credit has been assumed')
    incremental = (max(0, required-active_required) if active_required is not None else (None if resident else required)) if required is not None else None
    headroom = machine['available']-incremental if machine.get('available') is not None and incremental is not None else None
    cold = machine['available']-required if machine.get('available') is not None and required is not None else None
    minimum_reservation = int((values['reservation_gib'] or 0)*GiB)
    known_requirement = required if required is not None else minimum_reservation or None
    # Explicit configured caps are known even if restart availability is not.
    explicit_caps = [values[k]*GiB for k in ('budget_gib','ceiling_gib') if values[k] is not None]
    if known_requirement and explicit_caps and known_requirement > min(explicit_caps):
        warnings.append('Forecast reservation exceeds the explicit configured budget or ceiling' if required is not None else 'Minimum reservation alone exceeds the explicit budget or ceiling; full model memory is unknown')
    if policy and known_requirement and known_requirement > machine['total']-policy['preferred']:
        warnings.append('Forecast reservation exceeds physical RAM minus the admission reserve; unloading cannot make this configuration admissible')
    if policy and not resident:
        if known_requirement and not explicit_caps and known_requirement > min(policy['budget'], policy['ceiling']):
            warnings.append('Forecast exceeds the proposed next-admission budget at current availability')
        if known_requirement and machine['available']-known_requirement < policy['preferred']:
            warnings.append('Cold-load headroom is below the proposed admission reserve' if required is not None else 'Minimum reservation alone leaves less than the admission reserve; full model memory is unknown')
        if machine['available'] < policy['floor']: warnings.append('Current available memory is below the proposed runtime stop floor')
    if resident:
        info.append('The resident model is already included in current memory usage. This draft comparison shows the estimated increase above its active forecast, not a second copy of the model or an operation admission decision.')
        info.append('Headroom and automatic budget after unload/restart are unknown until memory is measured again. Recommendations remain bounded by known physical memory and explicit caps; fresh admission is still required.')
        if active_policy and machine.get('available') is not None:
            if machine['available'] < active_policy['floor']:
                warnings.append('Current available memory is below the active reservation runtime floor')
            elif incremental and headroom is not None and headroom < active_policy['preferred']:
                warnings.append('The estimated additional reservation leaves less than the active admission reserve')
    return_info = dict(info=info, explicit_cap=min(explicit_caps) if explicit_caps else None, policy_hypothetical=resident and values['budget_gib'] is None,
                       post_unload_remaining=None if resident else cold)
    return dict(machine=machine, forecast=estimate, required=required, minimum_reservation=minimum_reservation, active_reservation=active_required or None,
                resident=resident, incremental=incremental, remaining=headroom, cold_remaining=cold, policy=policy, active_policy=active_policy,
                active_operation=active_operation, active_static_cap=active_static_cap,
                backend=backend, effective_context=context, kernel_token_rows=NATIVE_ROWS[backend['selected']], prefill_chunk=PREFILL_CHUNK, model_context=model_context,
                errors=errors, warnings=warnings, **return_info,
                qualification='Advisory interactive-generation estimate: bounded prefill, full-history KV cache and last-token logits. Full trace uses a separate larger reservation. Each operation checks fresh available memory for its new buffers and fixes its measured-process limit only while it runs. Idle limits remain bounded by physical memory and explicit caps.')


def capacity_recommendation(settings, ws=None, telemetry=None, runtime=None, overrides=None):
    """Largest metadata-forecast window; never promises reclaimed RAM or admission."""
    from .memory_guard import snapshot, forecast
    from .runtime_supervisor import policy_for
    values = dict(settings['values'], budget_gib=None, ceiling_gib=None, reservation_gib=None)
    values.update(overrides or {})  # Retained launch caps still constrain a proposal.
    model_context = getattr(getattr(ws, 'struct', None), 'n_ctx', None)
    upper = model_context or DEFAULT_CONTEXT
    if (overrides or {}).get('context') is not None:
        upper = min(upper, overrides['context'])
    if type(upper) is not int or not 2 <= upper <= MAX_EXACT_INTEGER:
        return dict(changes=None, status='blocked', note='No usable generation window: model context must allow a prompt and a response.')
    capacity = dict(cap=None, margin=None, target=None)
    resident = bool(ws is not None and getattr(ws, 'runner_loaded', False))
    note = ('Resident recommendation is only a physical upper bound. Post-unload availability is unknown; fresh admission is required.' if resident else
            'Recommendation uses current measured availability and the conservative forecast; fresh admission is required.')
    if not model_context:
        note += f' Model window is unknown; the search uses the {DEFAULT_CONTEXT}-token fallback.'
    if (overrides or {}).get('context') is not None:
        note += ' The retained launch context override bounds this recommendation.'
    if ws is None or ws.kind != 'gguf':
        candidate = min(DEFAULT_CONTEXT, upper)
        note = 'Model memory/window is unknown or not covered. These fallback values are not a fit recommendation.'
        status = 'uncertain'
    else:
        try:
            machine = (telemetry or snapshot)()
            if not 0 < machine['available'] <= machine['total']: raise ValueError('Memory telemetry unavailable')
            if machine.get('pressure') != 1: raise ValueError('Memory pressure is not normal; fresh admission would be refused')
            policy = policy_for(values, machine['total'], machine['total'] if resident else machine['available'])
            cap = min(policy.budget, policy.ceiling, machine['total']-policy.preferred)
            if not resident: cap = min(cap, machine['available']-policy.preferred)
            margin = max(RECOMMENDATION_BUFFER_MIN, math.ceil(cap*RECOMMENDATION_BUFFER_FRACTION))
            target = cap-margin
            capacity = dict(cap=cap, margin=margin, target=target)
            buffer_note = f' Recommendation buffer: {margin/GiB:.2f} GiB (max of {RECOMMENDATION_BUFFER_MIN/GiB:g} GiB and {100*RECOMMENDATION_BUFFER_FRACTION:g}% of the {cap/GiB:.2f} GiB cap); forecast target {target/GiB:.2f} GiB. This preset-only slack leaves room for availability changes and small edits; it is not a new hard runtime limit or a physical reservation.'
            note += buffer_note
            if target <= 0:
                return dict(changes=None, status='blocked', note='The known memory cap cannot accommodate the recommendation buffer; no preset is applied.'+buffer_note, **capacity)
            def fits(context):
                # CLI/environment context remains authoritative on the next start.
                effective = context_limit((overrides or {}).get('context', context), model_context)
                return max(forecast(ws, effective, operation='interactive')['total'], int((values['reservation_gib'] or 0)*GiB)) <= target
            if not fits(2):
                return dict(changes=None, status='blocked', note=('The retained launch context cannot fit the buffered forecast target. Change the launch override before requesting a preset.' if (overrides or {}).get('context') else 'No context with response space fits the buffered forecast target, even at two tokens. Reduce the model or memory requirement; no preset is applied.')+buffer_note, **capacity)
            low, high = 2, upper
            while low < high:
                mid = (low+high+1)//2
                if fits(mid): low = mid
                else: high = mid-1
            candidate = low
            status = 'uncertain' if resident else 'estimated'
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return dict(changes=None, status='unknown', note='Cannot recommend capacity: '+str(exc))
    return dict(changes=dict(context=candidate, new_tokens=min(DEFAULT_OUTPUT, max(1,candidate//4)),
                            resident_seconds=14400, budget_gib=None, ceiling_gib=None, reservation_gib=None),
                status=status, note=note, **capacity)


def interactive_defaults(settings, ws=None, telemetry=None, runtime=None):
    return capacity_recommendation(settings, ws, telemetry, runtime)['changes']
