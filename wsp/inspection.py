"""Bounded single-flight cache for read-only inspection, never tensors or runners."""
from collections import OrderedDict
from concurrent.futures import Future
from threading import Lock, Event
from uuid import uuid4
from weakref import WeakKeyDictionary


class InspectionBusy(RuntimeError):
    pass


class InspectionRequests:
    def __init__(self, limit=64, max_pending=4):
        self.limit = limit
        self.max_pending = max_pending
        self.lock = Lock()
        self.identities = WeakKeyDictionary()
        self.pending = {}
        self.completed = OrderedDict()
        self.cancellations = {}

    def identity(self, workspace):
        with self.lock:
            if workspace not in self.identities:
                self.identities[workspace] = uuid4().hex
            return self.identities[workspace]

    def cancel(self, key):
        with self.lock:
            event = self.cancellations.get(key)
            if event is None: return False
            event.set()
            return True

    def run(self, key, calculate, cancellable=False):
        with self.lock:
            if key in self.completed:
                self.completed.move_to_end(key)
                return self.completed[key]
            future = self.pending.get(key)
            owner = future is None
            if owner:
                if len(self.pending) >= self.max_pending:
                    raise InspectionBusy("Other inspection calculations are still running. Try again when they finish.")
                future = self.pending[key] = Future()
                if cancellable: self.cancellations[key] = Event()
        if not owner:
            return future.result()
        try:
            result = calculate(self.cancellations[key]) if cancellable else calculate()
        except BaseException as exc:
            with self.lock:
                self.pending.pop(key, None)
                self.cancellations.pop(key, None)
                future.set_exception(exc)
            raise
        with self.lock:
            self.completed[key] = result
            while len(self.completed) > self.limit:
                self.completed.popitem(last=False)
            self.pending.pop(key, None)
            self.cancellations.pop(key, None)
            future.set_result(result)
        return result
