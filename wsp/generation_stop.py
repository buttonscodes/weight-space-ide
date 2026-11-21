"""Stopping a Run or Chat request when the person asks: a cooperative end, not an interruption.

A request that carries a ``stream`` id is registered in ``Stops`` while it runs;
``POST /api/generation/stop`` sets its ``Stop``. The route then

- starts no further variant (they stay ``not_run``), and
- ends the variant being generated at its next token: the route's token callback
  (``token_stream.TextStream`` given this ``Stop``) raises ``GenerationStopped``
  after recording a token once the stop is set.

Where that exception goes depends on the runner, and each place is the only one
that handles it:

- ``FunctionalRunner`` and an in-process ``GGUFRunner``: it leaves ``generate``;
  the route keeps the tokens the callback saw.
- ``HFRunner``: ``runner.TokenCallback`` catches it and, as a transformers
  stopping criterion, ends ``generate`` normally after that token.
- ``RemoteRunner``: its frame loop catches it (it is never a lost worker) and
  writes ``stop.json`` naming the job into the runtime directory. While the worker
  is still reading the prompt no token arrives, so the loop also looks at the
  callback's ``Stop`` each quiet second. The model worker reads that file at two
  places of a ``generate`` job: after each token it emits (``_job``'s ``emit``),
  and at its lease checks (``WorkerLease.require`` and ``layer_checkpoint``),
  which run between prompt chunks and layers. Either way it answers the job with
  an ordinary ``result`` frame holding the tokens generated so far, possibly none.

Nothing else changes: no lease is revoked, no runtime record or loss is written,
the worker keeps its loaded model and the next request runs at once.
"""
from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

FILE = 'stop.json'
REMEMBERED = 256      # ids kept of finished requests and of stops that came early


class GenerationStopped(Exception):
    """Raised by a token callback once the person asked to stop."""

    def __init__(self):
        super().__init__('Generation stopped on request')


class Stop:
    """One running request's stop flag."""

    def __init__(self):
        self._event = threading.Event()

    def set(self):
        self._event.set()

    def is_set(self):
        return self._event.is_set()


class Stops:
    """The running requests that can be stopped, by stream id.

    A stop can reach the server before its request does: the browser sends both
    at once, and a supervised server holds a request behind the one it is
    running, for as long as that one takes. A stop for an id that no request has
    used is therefore remembered, without a time limit (ids are fresh per request,
    and at most ``REMEMBERED`` are kept), and applies when that request arrives;
    it still answers ``found: false``. A stop for a request that has finished is
    not remembered.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.running = {}
        self.early, self.finished = OrderedDict(), OrderedDict()

    @staticmethod
    def _remember(ids, stream, value):
        ids.pop(stream, None)
        ids[stream] = value
        while len(ids) > REMEMBERED:
            ids.popitem(last=False)

    @contextmanager
    def open(self, stream):
        """Register ``stream`` (``None``: nothing) for the duration of one request. -> its Stop or None."""
        if stream is None:
            yield None
            return
        with self.lock:
            if stream in self.running:
                raise ValueError('This stream id is already used by a running request; send a new id with each request')
            stop = self.running[stream] = Stop()
            if self.early.pop(stream, None) is not None:
                stop.set()
        try:
            yield stop
        finally:
            with self.lock:
                self.running.pop(stream, None)
                self._remember(self.finished, stream, True)

    def get(self, stream):
        with self.lock:
            return self.running.get(stream)

    def stop(self, stream):
        """Ask the running request ``stream`` to stop. -> False when no running request has that id."""
        with self.lock:
            stop = self.running.get(stream)
            if stop is None:
                if stream not in self.finished:
                    self._remember(self.early, stream, True)
                return False
        stop.set()
        return True


# ------------------------------------------------------------------ server -> model worker
def request(directory, job):
    """Server side: ask the model worker to end ``job``. Never raises. -> written?"""
    try:
        path = Path(directory)/FILE
        temporary = path.with_name(f'{FILE}.{os.getpid()}.tmp')
        temporary.write_text(json.dumps({'job': str(job)}))
        temporary.replace(path)
        return True
    except Exception:
        return False


def requested(directory, job):
    """Worker side, after each emitted token: was ``job`` asked to stop? Never raises."""
    try:
        return json.loads((Path(directory)/FILE).read_text()).get('job') == job
    except Exception:
        return False


def clear(directory, job):
    """Server side, when the job has answered: remove its request. Never raises."""
    try:
        if requested(directory, job):
            (Path(directory)/FILE).unlink()
    except Exception:
        pass
