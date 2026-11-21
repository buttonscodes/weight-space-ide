"""Cached harness readiness and explicit sign-in. Never stores or exposes tokens."""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .native_harness import BRIDGE, billing_of

STATUS_TTL = 60
STATUS_TIMEOUT = 20
LOGIN_TIMEOUT = 600
ERROR = "Could not verify authentication. Check the installed CLI and try again."


def safe_url(provider, value):
    if not isinstance(value, str) or len(value) > 12000:
        return None
    try:
        u = urlparse(value)
        if u.scheme != 'https' or u.username or u.password or u.port or u.fragment:
            return None
        if provider == 'claude':
            q = parse_qs(u.query)
            if u.hostname not in ('claude.com', 'claude.ai') or u.path not in ('/cai/oauth/authorize', '/oauth/authorize'):
                return None
            if q.get('code') != ['true'] or q.get('redirect_uri') != ['https://platform.claude.com/oauth/code/callback']:
                return None
        elif provider == 'codex':
            if u.hostname not in ('auth.openai.com', 'chatgpt.com') or u.path not in ('/oauth/authorize', '/authorize'):
                return None
        else:
            return None
        return value
    except ValueError:
        return None


class AuthJob:
    def __init__(self, provider, operation, context):
        self.provider = provider
        self.operation = operation
        self.context = context
        self.id = secrets.token_hex(16)
        self.data = {'provider': provider, 'status': 'checking' if operation == 'status' else 'signing_in',
                     'attempt_id': self.id, 'needs_code': False}
        self.lock = threading.RLock()
        self.retired = threading.Event()
        self.spawned = threading.Event()
        self.cancelled = threading.Event()
        self.proc = None
        self.updated = time.monotonic()
        self.timeout = STATUS_TIMEOUT if operation == 'status' else LOGIN_TIMEOUT
        self.data['expires_at'] = time.time() + self.timeout
        self.timer = threading.Timer(self.timeout, self.expire)
        self.timer.daemon = True
        self._acks = {}
        self._write_lock = threading.Lock()

    def snapshot(self):
        with self.lock:
            return dict(self.data)

    def start(self):
        self.timer.start()
        threading.Thread(target=self.pump, daemon=True, name=f'agent-auth-{self.provider}').start()

    def write(self, value):
        with self._write_lock:
            if not self.proc or self.proc.poll() is not None:
                raise RuntimeError('Sign-in is no longer active')
            self.proc.stdin.write(json.dumps(value) + '\n')
            self.proc.stdin.flush()

    def accept(self, frame):
        with self.lock:
            if self.cancelled.is_set():
                return
            if 'id' in frame:
                ack = self._acks.get(frame['id'])
                if ack:
                    ack['ok'] = frame.get('ok') is True
                    ack['done'].set()
                return
            status = frame.get('status')
            if status not in ('checking', 'ready', 'signed_out', 'signing_in', 'verifying', 'error'):
                return
            self.data['status'] = status
            self.data['needs_code'] = frame.get('needs_code') is True
            if status in ('ready', 'signed_out', 'error'):
                self.data.pop('url', None)
                self.data['needs_code'] = False
            if status == 'error':
                self.data['message'] = ERROR
            if status == 'ready':
                method = frame.get('method')
                self.data['method'] = method if method in ('claude.ai', 'oauth_token', 'api_key', 'auth_token', 'bedrock', 'vertex', 'foundry', 'none', 'configured', 'apiKey', 'chatgpt', 'amazonBedrock', 'gcp', 'configured provider') else 'configured'
                self.data['billing'] = billing_of(self.provider, self.data['method'])
                self.data['message'] = 'Authentication is configured. The provider validates it when a request is made.'
            if frame.get('url'):
                url = safe_url(self.provider, frame['url'])
                if not url:
                    raise RuntimeError('Unsupported sign-in URL')
                self.data['url'] = url

    def pump(self):
        try:
            executable, cwd = self.context
            self.proc = subprocess.Popen([shutil.which('node') or 'node', str(BRIDGE / 'auth.mjs')], cwd=cwd,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True, bufsize=1, start_new_session=True)
            self.spawned.set()
            if self.cancelled.is_set():
                return
            self.write({'method': 'start', 'params': {'provider': self.provider, 'operation': self.operation,
                                                    'cwd': cwd, 'executable': executable}})
            for line in self.proc.stdout:
                self.accept(json.loads(line))
            self.proc.wait()
            with self.lock:
                if not self.cancelled.is_set() and self.data['status'] not in ('ready', 'signed_out', 'error'):
                    self.data.update(status='error', message=ERROR, needs_code=False)
                    self.data.pop('url', None)
        except Exception:
            with self.lock:
                if not self.cancelled.is_set():
                    self.data.update(status='error', message=ERROR, needs_code=False)
                    self.data.pop('url', None)
        finally:
            self.spawned.set()
            self.timer.cancel()
            self.kill_group()
            with self.lock:
                for ack in self._acks.values():
                    ack['done'].set()
            self.updated = time.monotonic()
            self.retired.set()

    def kill_group(self):
        if self.proc:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def expire(self):
        self.cancel('error', 'Sign-in timed out. Try again.' if self.operation == 'login' else 'Authentication check timed out. Retry the check.')

    def cancel(self, status='cancelled', message='Sign-in cancelled.'):
        self.cancelled.set()
        self.timer.cancel()
        with self.lock:
            self.data.update(status=status, message=message, needs_code=False)
            self.data.pop('url', None)
        self.spawned.wait(2)
        if self.proc and self.proc.poll() is None:
            try:
                self.write({'method': 'cancel'})
                self.proc.wait(timeout=2)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                pass
        self.kill_group()
        # The pump owns retirement, including a late spawn after cancellation.
        self.retired.wait(2)
        return self.snapshot()

    def code(self, value):
        if not isinstance(value, str) or len(value) > 4096 or not re.fullmatch(r'[A-Za-z0-9._~+/=\-]+(?:#[A-Za-z0-9._~+/=\-]+)?', value):
            raise ValueError('Invalid sign-in code')
        with self.lock:
            if self.retired.is_set() or self.cancelled.is_set() or not self.data.get('needs_code'):
                raise RuntimeError('Sign-in is not waiting for a code')
            self.data['needs_code'] = False
            self.data['status'] = 'verifying'
            rid = secrets.token_hex(8)
            ack = {'done': threading.Event(), 'ok': False}
            self._acks[rid] = ack
        try:
            try:
                self.write({'method': 'code', 'id': rid, 'code': value})
            except (OSError, RuntimeError):
                self.cancel('error', 'The sign-in code could not be submitted. Start sign-in again.')
                raise RuntimeError('The sign-in code could not be submitted. Start sign-in again.') from None
            if not ack['done'].wait(5) or not ack['ok']:
                self.cancel('error', 'The sign-in code could not be submitted. Start sign-in again.')
                raise RuntimeError('The sign-in code could not be submitted. Start sign-in again.')
            return self.snapshot()
        finally:
            with self.lock:
                self._acks.pop(rid, None)


class AuthManager:
    def __init__(self, sess, job_factory=AuthJob):
        self.sess = sess
        self.jobs = {}
        self.lock = threading.RLock()
        self.csrf = secrets.token_urlsafe(32)
        self.job_factory = job_factory
        self.closed = False

    def context(self, provider):
        if provider not in ('claude', 'codex'):
            raise ValueError('Unsupported authentication provider')
        executable = shutil.which(provider)
        if not executable or not shutil.which('node'):
            raise RuntimeError('Install the agent CLI and Node.js before checking authentication.')
        cwd = os.path.join(self.sess.ws.dir, 'agent') if self.sess.ws is not None else os.getcwd()
        os.makedirs(cwd, exist_ok=True)
        return executable, os.path.realpath(cwd)

    def status(self, provider, refresh=False):
        try:
            context = self.context(provider)
        except (ValueError, RuntimeError) as exc:
            return {'provider': provider, 'status': 'error', 'message': str(exc)}
        with self.lock:
            if self.closed:
                raise RuntimeError('Authentication manager is closed')
            job = self.jobs.get(provider)
            if job and not job.retired.is_set():
                if job.context == context:
                    return job.snapshot()
                job.cancel()
                if not job.retired.is_set():
                    raise RuntimeError('Previous authentication check is still closing')
            if not job or job.context != context or refresh or time.monotonic() - job.updated > STATUS_TTL:
                job = self.job_factory(provider, 'status', context)
                self.jobs[provider] = job
                job.start()
            return job.snapshot()

    def check(self, provider):
        status = self.status(provider, refresh=True)
        with self.lock:
            job = self.jobs.get(provider)
        if job and job.operation == 'status' and status.get('attempt_id') == job.id:
            if not job.retired.wait(STATUS_TIMEOUT + 4):
                job.cancel('error', 'Authentication check timed out.')
            return job.snapshot()
        return status

    def ensure_ready(self, provider):
        """The fresh readiness state (its `method` and `billing` say how requests will be paid for)."""
        state = self.check(provider)
        if state['status'] != 'ready':
            raise RuntimeError('Sign in to ' + ('Claude Code' if provider == 'claude' else 'Codex') + ' in the Programmer tab before starting.')
        return state

    def assert_no_login(self):
        with self.lock:
            if any(job.operation == 'login' and not job.retired.is_set() for job in self.jobs.values()):
                raise RuntimeError('Finish or cancel sign-in before changing workspace/task or starting an agent.')

    def login(self, provider):
        with self.lock:
            previous = self.jobs.get(provider)
            if previous and previous.operation == 'login' and not previous.retired.is_set():
                return previous.snapshot()
        state = self.check(provider)
        if state['status'] == 'ready':
            return state  # Preserve an existing account; never start replacement login.
        if state['status'] not in ('signed_out', 'cancelled'):
            raise RuntimeError('Authentication status is unknown. Retry the check before signing in.')
        with self.lock:
            self.assert_no_login()
            job = self.job_factory(provider, 'login', self.context(provider))
            self.jobs[provider] = job
            job.start()
            return job.snapshot()

    def active_attempt(self, provider, attempt_id):
        with self.lock:
            job = self.jobs.get(provider)
            if not job or job.operation != 'login' or job.id != attempt_id or job.retired.is_set():
                raise RuntimeError('This sign-in attempt is no longer active')
            return job

    def shutdown(self):
        with self.lock:
            self.closed = True
            jobs = list(self.jobs.values())
        for job in jobs:
            if not job.retired.is_set():
                job.cancel()
