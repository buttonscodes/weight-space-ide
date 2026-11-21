"""IDE-owned shutdown, before Uvicorn drains long-lived responses.

Uvicorn 0.54 Server.shutdown drains connections before lifespan.shutdown and skips
lifespan on force_exit. Keep those upstream semantics, but stop SSE first and always
finish our owned cleanup, including the second-SIGINT path.
"""
import asyncio
import logging
import signal
import threading

import uvicorn


log = logging.getLogger('uvicorn.error')


class Lifecycle:
    def __init__(self, resources, timeout=15):
        self.resources = resources  # (name, bounded close callback, owned Popen getter)
        self.timeout = timeout
        self.stopping = threading.Event()
        self.completed = False
        self.errors = []
        self._cleanup = None

    def begin_shutdown(self):
        self.stopping.set()

    async def cleanup(self):
        self.begin_shutdown()
        if self._cleanup is None:
            self._cleanup = asyncio.create_task(self._close())
        await asyncio.shield(self._cleanup)

    async def _close(self):
        # Independent resources close concurrently, so multiple agent sessions do
        # not multiply the shutdown deadline. Daemon workers avoid an unbounded
        # asyncio default-executor join if a faulty callback gets stuck.
        async def close_one(name, callback, processes):
            owned = {}
            def capture():
                for proc in processes():
                    if proc is None or proc.pid in owned or proc.poll() is not None:
                        continue
                    import psutil
                    try:
                        p = psutil.Process(proc.pid)
                        identities = {p.pid: p.create_time()}
                        for child in p.children(recursive=True):
                            identities[child.pid] = child.create_time()
                        owned[proc.pid] = (proc, identities)
                    except psutil.NoSuchProcess:
                        pass
            def terminate():
                capture()
                errors = []
                remaining = []
                for proc, identities in list(owned.values()):
                    from .memory_guard import stop_group
                    remaining.extend(stop_group(proc, identities, errors))
                if errors or remaining:
                    raise RuntimeError(f'owned cleanup errors={errors}, remaining={remaining}')
            try:
                capture()
            except Exception as exc:
                self.errors.append(f'{name} ownership capture: {type(exc).__name__}: {exc}')
            try:
                await _worker(callback, self.timeout)
            except Exception as exc:
                self.errors.append(f'{name}: {type(exc).__name__}: {exc}')
            finally:
                # Also reap a stubborn provider/legacy child after its close
                # callback failed or timed out. Only captured owned identities.
                try:
                    await _worker(terminate, 10)
                except Exception as exc:
                    self.errors.append(f'{name} process teardown: {type(exc).__name__}: {exc}')
        results = await asyncio.gather(*(close_one(*r) for r in self.resources()), return_exceptions=True)
        self.errors.extend(f'cleanup: {type(r).__name__}: {r}' for r in results if isinstance(r, BaseException))
        self.completed = True
        for error in self.errors:
            log.error('IDE shutdown incomplete: %s', error)


async def _worker(callback, timeout):
    loop = asyncio.get_running_loop()
    result = loop.create_future()
    def deliver(value, error):
        if not result.done():
            if error is not None:
                result.set_exception(error)
            else:
                result.set_result(value)
    def run():
        try:
            value, error = callback(), None
        except Exception as exc:
            value, error = None, exc
        try:
            loop.call_soon_threadsafe(deliver, value, error)
        except RuntimeError:  # loop already closed after a recorded timeout
            pass
    threading.Thread(target=run, name='ide-resource-close', daemon=True).start()
    return await asyncio.wait_for(result, timeout)


class IDEServer(uvicorn.Server):
    def __init__(self, config):
        super().__init__(config)
        self.ide_lifecycle = config.app.state.ide_lifecycle
        # Bound other stalled HTTP work too; SSE normally ends within100ms.
        if config.timeout_graceful_shutdown is None:
            config.timeout_graceful_shutdown = 3

    def handle_exit(self, sig, frame):
        self.ide_lifecycle.begin_shutdown()
        super().handle_exit(sig, frame)

    async def shutdown(self, sockets=None):
        self.ide_lifecycle.begin_shutdown()
        try:
            await super().shutdown(sockets=sockets)
            if self.force_exit and not self.ide_lifecycle.completed:
                # Upstream skips lifespan on the second SIGINT. It must still
                # receive shutdown, rather than being cancelled by asyncio.run.
                await self.lifespan.shutdown()
        finally:
            await self.ide_lifecycle.cleanup()


def serve(app, *, sockets=None, **config):
    server = IDEServer(uvicorn.Config(app, **config))
    try:
        server.run(sockets=sockets)
    except KeyboardInterrupt:
        # Uvicorn replays captured SIGINT after its shutdown completes. Only that
        # expected CLI interrupt is quiet; startup faults and cleanup errors are
        # never hidden, nor are arbitrary CancelledError exceptions filtered.
        if not (server.should_exit and server.ide_lifecycle.completed
                and signal.SIGINT in getattr(server, "_captured_signals", [])):
            raise
    if server.ide_lifecycle.errors:
        raise RuntimeError('IDE shutdown incomplete; see cleanup errors above')
