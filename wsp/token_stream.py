"""Live text of a running generation: token callbacks become ``generation`` events.

Run and Chat pass a ``TextStream`` as the runner's ``on_token``. Each event tells
the browser how much of the text it already shows is still valid (``keep``) and
what to append (``text``), so a retraction (a stop string that completes, a
character that needed one more token) needs no special case. Offsets count UTF-16
units, the unit of a JavaScript string.

Text events are throttled to one per ``INTERVAL``. A token that arrives inside
the interval is shown when the interval ends, by a timer, so text is never held
back while the model stalls.

The callback never raises, with one exception: given a ``stop`` (a
``generation_stop.Stop``) that the person has set, it raises
``GenerationStopped`` after recording the token. In the server it runs inside
``RemoteRunner``'s frame loop, where any other exception would be handled as a
lost model worker; that loop and ``runner.TokenCallback`` recognise this one.
"""
import re
import threading
import time

from .generation_stop import GenerationStopped

INTERVAL = 0.05          # seconds between text events; a slower model sends one per token
STREAM_ID = re.compile(r'[A-Za-z0-9_-]{1,64}\Z')
REPLACEMENT = '�'   # what an incomplete multi-byte character decodes to


def stream_id(body):
    """The request's optional ``stream`` id, validated. Absent means no live events."""
    value = body.get('stream')
    if value is None or value is False:
        return None
    if not isinstance(value, str) or not STREAM_ID.match(value):
        raise ValueError('stream must be 1-64 characters from A-Z, a-z, 0-9, "_" and "-"')
    return value


def _units(text):
    return len(text.encode('utf-16-le', 'surrogatepass'))//2


class TextStream:
    """``on_token`` for one variant of one request.

    ``render(ids) -> str`` must be the same function that produces the final text,
    so the live text converges on the response. With ``stop``, ``stopped`` says
    whether the callback raised ``GenerationStopped``; ``ids`` are the tokens it saw.
    """
    def __init__(self, publish, stream, variant, render, *, stop=None, interval=None, clock=time.monotonic, timer=threading.Timer):
        self.publish, self.stream, self.variant, self.render = publish, stream, variant, render
        self.stop, self.stopped = stop, False
        self.interval, self.clock, self.timer = INTERVAL if interval is None else interval, clock, timer
        self.ids, self.sent, self.last, self.failed = [], '', None, False
        # The lock orders the events of the request thread and of the timer thread.
        self.lock, self.pending, self.closed = threading.Lock(), None, False

    def _event(self, **fields):
        self.publish(dict(type='generation', stream=self.stream, variant=self.variant, **fields))

    def _flush(self, text, tokens):
        keep = 0
        for a, b in zip(self.sent, text):
            if a != b:
                break
            keep += 1
        if keep == len(self.sent) == len(text):
            return
        self._event(state='started', keep=_units(text[:keep]), text=text[keep:], tokens=tokens)
        self.sent = text

    def _fail(self):
        # The browser stops trusting its live text for this variant; the response still carries the result.
        self.failed = True
        try:
            self._event(state='started', lost=True)
        except Exception:
            pass

    def begin(self):
        try:
            self._event(state='started', keep=0, text='', tokens=0)
        except Exception:
            self._fail()

    def _show(self):
        self.last = self.clock()
        # A character still missing its last token decodes to U+FFFD: hold it back.
        self._flush(self.render(list(self.ids)).rstrip(REPLACEMENT), len(self.ids))

    def __call__(self, token):
        try:
            with self.lock:
                self.ids.append(int(token))
                if self.failed or self.closed:
                    return
                wait = 0 if self.last is None else self.interval-(self.clock()-self.last)
                if wait <= 0:
                    self._show()
                elif self.pending is None:
                    self.pending = self.timer(wait, self._late)
                    self.pending.daemon = True
                    self.pending.start()
        except Exception:
            self._fail()
        if self.stop is not None and self.stop.is_set():
            self.stopped = True
            raise GenerationStopped()

    def _late(self):
        try:
            with self.lock:
                self.pending = None
                if not (self.failed or self.closed):
                    self._show()
        except Exception:
            self._fail()

    def close(self):
        """No more live text: after this only ``end`` publishes. Never raises."""
        try:
            with self.lock:
                self.closed = True
                if self.pending is not None:
                    self.pending.cancel()
                    self.pending = None
        except Exception:
            self.failed = True

    def end(self, state, ids=None, text=None):
        """The variant finished: its final text (``text``, else rendered from ``ids``), then ``state``."""
        self.close()
        ids = self.ids if ids is None else ids
        try:
            if not self.failed:
                with self.lock:
                    self._flush(self.render(list(ids)) if text is None else text, len(ids))
        except Exception:
            self._fail()
        try:
            self._event(state=state, tokens=len(ids), **(dict(lost=True) if self.failed else {}))
        except Exception:
            self.failed = True
