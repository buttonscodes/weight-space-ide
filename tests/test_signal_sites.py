"""Static enumeration of every signal the runtime supervisor can send (no processes are started).

A new signal path needs a reviewed edit of ``SIGNAL_SITES`` and of the pins
below. ``violations(source)`` is the checker; the negative tests feed it the
real module with one bypass inserted and require each to be rejected. It is a
tripwire for accidental regressions and for the evasions listed in those tests,
not a proof against deliberate obfuscation. What it enforces, over
``wsp/runtime_supervisor.py``:

- raw signal calls, attribute access to signal methods (``kill``,
  ``terminate``, ``send_signal``, ``killpg``, ``pthread_kill``, ``stop_group``,
  ``kill_group``, for example a psutil process's ``.kill()``) and references to
  ``stop_group``/``kill_group`` occur only inside ``_signal_server`` and
  ``_signal_worker``;
- no ``getattr`` with a dynamic or signal-method name, no call taking a
  signal-method name as a literal, no ``eval``/``exec``/``__import__``; no
  aliased or star import of a signal helper; no ``os.system``/``os.popen``/
  ``os.exec*``/``os.spawn*``; ``subprocess`` only as ``Popen`` of the pinned
  ``wsp serve`` and ``wsp worker`` command lists; no ``kill``/``pkill`` command
  string anywhere;
- every helper call names a literal reason whose ``SIGNAL_SITES`` target is
  the helper's; helpers and the ``forward`` handler are never aliased, and
  ``forward`` is only installed with ``signal.signal``;
- each ``_signal_server`` call is pinned by its enclosing function and guard
  path (``SERVER_SITES``): the user-signal ``forward`` handler, the post-spawn
  ``stopping`` guard, and the teardown ``finally`` (forced shutdown, user
  shutdown, sweep of an exited server). The server reasons are exactly
  ``user_shutdown``, ``shutdown_forced`` and ``server_exited_cleanup``;
- ``_signal_worker`` never names the server (``child``/``known``) and
  ``_signal_server`` never names the worker; the worker record only ever holds
  the ``Popen`` of the worker command, and its identity set only grows from
  the worker tree's own sample: a record's ``known``/``identities`` entry is
  never assigned, augmented (``|=``) or item-stored, and no method is called
  on it except ``worker_tree``'s ``w['known'].update(tree['identities'])``;
- ``os``, ``signal``, ``subprocess`` and ``psutil`` are imported only as the
  real file does (``PROCESS_IMPORTS``: plain ``import``, no second import);
  no from-import of those or of ``asyncio``/``multiprocessing``/``ctypes``,
  and no aliased import except ``from . import runtime_policy as rp``;
- inside ``supervise`` (``STATE_SITES``, ``EXIT_SITES``): ``stopping`` and
  ``forced`` are stored only at their pinned sites (``nonlocal`` only in
  ``forward``); no ``signal.raise_signal``/``alarm``/``setitimer``,
  ``sys.exit``, ``os._exit``, ``os.abort``, ``exit``/``quit`` or ``raise`` of
  ``SystemExit``/``KeyboardInterrupt``/``GeneratorExit``/``BaseException``, under
  any spelling; ``signal.signal`` only at the three pinned install/restore
  sites; ``break`` and ``return`` only at their pinned sites; and a ``raise``
  leaves a supervision loop only at a pinned site (one inside the body of a
  ``try`` with an ``except Exception`` handler within the loop is contained).
- ``bindings``: the server ``Popen`` (``child``) and the saved handlers
  (``handlers``) are bound only at their pinned sites (``ROOT_SITES``) under
  any binding form; ``child`` is only read for ``pid``/``poll``/
  ``returncode``/``send_signal``, compared with ``None`` or stopped by the
  server helper (never aliased or passed on); ``handlers`` is read only by the
  install and the two restores (``HANDLER_USES``), never called through, and a
  saved ``handler`` only goes back to ``signal.signal``; nothing inside
  ``supervise`` stores an attribute and nothing in the module writes ``Popen``
  state (``returncode``, ``pid``, ``poll``, ...); ``suspend``/``resume``/
  ``pidfd_send_signal`` are signal methods; ``signal.getsignal``,
  ``default_int_handler``, ``setattr``/``delattr``/``vars``/``globals``/
  ``locals``, frame and closure access, dunder attributes other than
  ``__name__``, ``sys.modules`` and imports of ``ctypes``/``cffi``/
  ``builtins``/``inspect``/``gc``/``importlib``/``operator``/``_thread``/
  ``multiprocessing`` are banned module-wide; ``getattr`` never names a
  signal, exit, reflection or ``Popen``-state attribute; the exit calls are
  banned module-wide (so a module-level helper cannot end supervision either);
  and an exception that escapes ``except Exception`` (``SystemExit``,
  ``KeyboardInterrupt``, ``GeneratorExit``, ``BaseException``,
  ``BaseExceptionGroup``, ``CancelledError``) is referenced, as a name,
  attribute or string, only at the spawn ``except BaseException`` that
  restores the handlers and re-raises.

Across ``wsp/`` (``test_group_stop_helpers_are_reachable_only_from_the_two_helpers``):
``memory_guard.stop_group``/``kill_group`` are called only by the two helpers,
by the standalone outer acceptance guard ``memory_guard.supervise`` (``python
-m wsp.memory_guard``, a separate program that owns its own child) and by
``lifecycle``'s teardown of the IDE server's own agent-provider processes
(which runs in the server, not the supervisor).

Not covered (would need a runtime sandbox): code in another module that the
supervisor calls without naming a signal method here (a new helper in
``memory_guard``, ``psutil`` internals, a ``ctypes``/``cffi`` call or native
extension reached through an object or module another import already holds,
for example a function in another ``wsp`` module that builds
``ctypes.CDLL(None)['ki'+'ll']`` from concatenated strings); process creation
through other libraries; indirect exits through library calls that raise
``SystemExit`` (for example ``argparse.ArgumentParser().error(...)`` or
``unittest.main(...)``); a signal sent to the supervisor itself by such a
route; an exception class from another module that subclasses
``BaseException`` under a different name; an exception object chosen at run
time (``raise exc``), which is not typed statically; identity sets or other
objects (``known``, a ``psutil.Process``) changed through an alias of the
record entry; and the server's own ``known`` set changed through methods.
Raises inside the nested helpers that the loops call (``poll_worker``,
``identity``, ``start_worker``, ...) are not pinned: an ``Exception`` reaches
the terminal ``watchdog_unavailable`` path, which never signals the server.
"""
import ast
import re
from pathlib import Path

import pytest

from wsp.runtime_supervisor import SIGNAL_SITES

APP = Path(__file__).resolve().parents[1]
SOURCE = APP/'wsp'/'runtime_supervisor.py'
SIGNAL_NAMES = {'send_signal', 'kill', 'terminate', 'killpg', 'pthread_kill', 'pidfd_send_signal', 'suspend', 'resume',
                'stop_group', 'kill_group'}
GROUP_HELPERS = {'stop_group', 'kill_group'}
HELPERS = {'_signal_server': 'server', '_signal_worker': 'worker'}
SERVER_REASONS = {'user_shutdown', 'shutdown_forced', 'server_exited_cleanup'}
WORKER_REASONS = {'kill_worker', 'worker_exited', 'release_reap', 'spawn_rollback', 'watchdog_failed', 'teardown',
                  'sampling_failed'}
# Every _signal_server call: (enclosing function, guard path from it, reason). Exactly these, once each.
SERVER_SITES = sorted([
    ('forward', (), 'user_shutdown'),                                               # user SIGINT/SIGTERM
    ('supervise', ('try', 'if stopping is not None'), 'user_shutdown'),             # a user signal during spawn
    ('supervise', ('finally', 'if forced or stopping is not None or child.poll() is not None', 'try',
                   'if forced'), 'shutdown_forced'),                                # user shutdown past 30 s
    ('supervise', ('finally', 'if forced or stopping is not None or child.poll() is not None', 'try',
                   'else of forced', 'if stopping is not None'), 'user_shutdown'),
    ('supervise', ('finally', 'if forced or stopping is not None or child.poll() is not None', 'try',
                   'else of forced', 'else of stopping is not None'), 'server_exited_cleanup'),
])
PROCESS_CALLS = re.compile(r'^os\.(system|popen|exec\w*|spawn\w*|posix_spawn\w*|fork\w*)$|^(pty|commands)\.')
KILL_COMMAND = re.compile(r'(^|[\s/;&|])(p?kill|killall)(\s|$)')
COMMANDS = {'command': "sys.executable, '-m', 'wsp', 'serve'", 'argv': "sys.executable, '-m', 'wsp', 'worker'"}
SERVER_NAMES = {'child', 'known', 'command'}
WORKER_NAMES = {'worker', 'w', 'spawned', 'proc', 'argv', 'worker_identity'}
LOOP = 'while child.poll() is None'
GRACE = ('if now >= next_sample', 'except', 'ExceptHandler',
         'if now - sampling_failed_since >= TELEMETRY_FAILURE_GRACE_SECONDS', 'if not measuring')
# Inside supervise: every store of the shutdown state, as (name, function, guard path).
STATE_SITES = sorted([
    ('stopping', 'supervise', ()), ('forced', 'supervise', ()),                     # initial values
    ('stopping', 'forward', ('if stopping is None',)),                              # the user signal
    ('forced', 'supervise', ('try', LOOP, 'if stopping is not None and now - stopping > 30')),
    ('forced', 'supervise', ('except', 'ExceptHandler', LOOP, 'if stopping is not None and now - stopping > 30')),
])
# Inside supervise (not its helpers): the only ways out of supervision.
EXIT_SITES = sorted([
    ('break', ('try', LOOP, 'if stopping is not None and now - stopping > 30')),
    ('break', ('except', 'ExceptHandler', LOOP, 'if stopping is not None and now - stopping > 30')),
    ('return', ()),
    ('raise', ('except', 'ExceptHandler')),                                         # spawn failed
    ('raise', ('try', 'except', 'ExceptHandler', 'if child.poll() is None')),         # start-up identity
    ('raise', ('try', LOOP, *GRACE)),                                              # a supervisor fault past grace
])
SIGNAL_INSTALLS = sorted([
    ('try', 'for (signal.SIGINT, signal.SIGTERM)'),
    ('except', 'ExceptHandler', 'for handlers.items()'),
    ('finally', 'for handlers.items()'),
])
EXIT_CALLS = {'raise_signal', 'alarm', 'setitimer', '_exit', 'abort', 'exit', 'quit', 'interrupt_main'}
EXIT_EXCEPTIONS = {'SystemExit', 'KeyboardInterrupt', 'GeneratorExit', 'BaseException', 'BaseExceptionGroup',
                   'CancelledError'}
BROAD = {'Exception', 'BaseException'}
# Reaching a handler, a closure cell or an object's state without naming it: never in the module.
REFLECTION = {'getsignal', 'default_int_handler', 'setattr', 'delattr', 'vars', 'globals', 'locals',
              'cell_contents', 'f_locals', 'f_globals', 'f_back', '_getframe', 'currentframe', 'get_referrers'}
BANNED_MODULES = {'ctypes', '_ctypes', 'cffi', 'builtins', 'inspect', 'gc', 'importlib', 'operator', '_thread',
                  '_signal', 'posix', 'multiprocessing', 'pty'}
# Where the server Popen and the saved handlers are bound (name, function, guard path); nowhere else.
ROOT_SITES = sorted([
    ('child', 'supervise', ()), ('handlers', 'supervise', ()),                    # child=None;known={};handlers={}
    ('child', 'supervise', ('try',)),                                             # child=subprocess.Popen(command,...)
])
# Every read of handlers: the install store and the two restores. Never called through.
HANDLER_USES = sorted([
    ('install', ('try', 'for (signal.SIGINT, signal.SIGTERM)')),
    ('restore', ('except', 'ExceptHandler', 'for handlers.items()')),
    ('restore', ('finally', 'for handlers.items()')),
])
# Every reference to an exception that escapes ``except Exception`` (as a name, attribute or string).
EXIT_EXCEPTION_SITES = [('BaseException', 'supervise', ('except', 'ExceptHandler'))]  # spawn failed: restore, re-raise
CHILD_ATTRIBUTES = {'pid', 'poll', 'returncode', 'send_signal'}                     # the reads a server Popen gets
POPEN_STATE = {'returncode', 'pid', 'poll', 'wait', 'send_signal', 'terminate', 'kill', 'suspend', 'resume'}
# Process modules are imported only in the real file's forms (module, function): plain, never aliased or from-imported.
PROCESS_MODULES = {'os', 'signal', 'subprocess', 'psutil', 'asyncio', 'multiprocessing', 'ctypes'}
PROCESS_IMPORTS = sorted([('os', '<module>'), ('signal', '<module>'), ('subprocess', '<module>'), ('psutil', 'supervise')])
IMPORT_ALIASES = [('runtime_policy', 'rp', 'supervise')]                          # the one aliased import: from . import runtime_policy as rp
# The worker record's identity sets: changed only by worker_tree's update from the worker tree's own sample.
IDENTITY_KEYS = {'known', 'identities'}


# ------------------------------------------------------------------ the checker
def _parents(module):
    parents = {}
    for node in ast.walk(module):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _field(parent, child):
    for name, value in ast.iter_fields(parent):
        if value is child or (isinstance(value, list) and any(v is child for v in value)):
            return name
    return None


def _step(parent, child):
    field = _field(parent, child)
    if isinstance(parent, ast.If):
        test = ast.unparse(parent.test)
        return {'body': f'if {test}', 'orelse': f'else of {test}'}.get(field)
    if isinstance(parent, ast.Try):
        return {'body': 'try', 'finalbody': 'finally', 'handlers': 'except', 'orelse': 'try-else'}[field]
    if isinstance(parent, ast.IfExp):
        test = ast.unparse(parent.test)
        return {'body': f'if {test}', 'orelse': f'else of {test}'}.get(field)
    if isinstance(parent, (ast.While, ast.For)):
        return f'{type(parent).__name__.lower()} {ast.unparse(parent.test if isinstance(parent, ast.While) else parent.iter)}'
    if isinstance(parent, (ast.With, ast.BoolOp, ast.Lambda, ast.comprehension, ast.ExceptHandler)):
        return type(parent).__name__
    return None


def site(node, parents):
    """(innermost enclosing function, guard path from it to ``node``)."""
    path, child = [], node
    parent = parents.get(id(child))
    while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
        step = _step(parent, child)
        if step:path.append(step)
        child, parent = parent, parents.get(id(parent))
    return (parent.name if parent is not None else '<module>'), tuple(reversed(path))


def called(node):
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None


def dotted(node):
    try:
        return ast.unparse(node.func)
    except Exception:
        return ''


def violations(source):
    """Every way ``source`` could signal a process outside the reviewed sites; [] when clean."""
    module = ast.parse(source)
    parents = _parents(module)
    found = []
    defs = {}
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.setdefault(node.name, []).append(node)
    for name in (*HELPERS, 'forward', 'start_worker', 'worker_tree'):
        if len(defs.get(name, ())) != 1:
            found.append(f'{name} must be defined exactly once, found {len(defs.get(name, ()))}')
    if found:
        return found
    inside = {name: {id(n) for n in ast.walk(defs[name][0])} for name in HELPERS}
    helper_nodes = inside['_signal_server'] | inside['_signal_worker']
    supervise = next((d for d in defs.get('supervise', ())), None)
    in_supervise = {id(n) for n in ast.walk(supervise)} if supervise else set()

    for node in ast.walk(module):
        where = f'line {getattr(node, "lineno", "?")}'
        outside = id(node) not in helper_nodes
        # Raw signal calls and any access to a signal method outside the helpers.
        if isinstance(node, ast.Attribute) and node.attr in SIGNAL_NAMES and outside:
            found.append(f'{where}: .{node.attr} outside the signal helpers')
        if isinstance(node, ast.Name) and node.id in SIGNAL_NAMES and outside and isinstance(node.ctx, ast.Load):
            found.append(f'{where}: {node.id} referenced outside the signal helpers')
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                base = alias.name.rsplit('.', 1)[-1]
                if alias.name == '*':
                    found.append(f'{where}: star import from {node.module}')
                elif base in SIGNAL_NAMES:
                    allowed = (isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == 'memory_guard'
                               and base in GROUP_HELPERS and alias.asname is None and id(node) in in_supervise)
                    if not allowed:
                        found.append(f'{where}: import of {alias.name}' + (f' as {alias.asname}' if alias.asname else ''))
        if not isinstance(node, ast.Call):
            continue
        name, full = called(node), dotted(node)
        # Reflection and dynamic code.
        if name in ('eval', 'exec', 'compile', '__import__', 'import_module', 'methodcaller', 'attrgetter'):
            found.append(f'{where}: {name}() can hide a signal call')
        if name == 'getattr' and (len(node.args) < 2 or not isinstance(node.args[1], ast.Constant)):
            found.append(f'{where}: getattr with a computed attribute name')
        literals = [*node.args, *(k.value for k in node.keywords)]
        if outside and any(isinstance(a, ast.Constant) and a.value in SIGNAL_NAMES for a in literals):
            found.append(f'{where}: {full}() takes a signal method name')
        # Shell and process spawning.
        if PROCESS_CALLS.match(full):
            found.append(f'{where}: {full}() can run kill')
        if full.startswith('subprocess.') or name in ('Popen', 'run', 'call', 'check_call', 'check_output'):
            arg = node.args[0] if node.args else None
            if full != 'subprocess.Popen' or not isinstance(arg, ast.Name) or arg.id not in COMMANDS:
                found.append(f'{where}: {full}() is not a pinned Popen of the server or worker command')
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and KILL_COMMAND.search(sub.value):
                found.append(f'{where}: {full}() mentions a kill command ({sub.value!r})')

    # The pinned command lists: every assignment starts with the wsp serve / wsp worker prefix.
    for node in ast.walk(module):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for t in ast.walk(target):
                    if isinstance(t, ast.Name) and t.id in COMMANDS and id(node) in in_supervise:
                        value = getattr(node, 'value', None)
                        if not (isinstance(node, ast.Assign) and isinstance(value, ast.List)
                                and ast.unparse(value).startswith('[' + COMMANDS[t.id])):
                            found.append(f'line {node.lineno}: {t.id} is not the pinned {COMMANDS[t.id]} command')
    # No kill command string anywhere outside a subscript key.
    for node in ast.walk(module):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and KILL_COMMAND.search(node.value):
            parent = parents.get(id(node))
            # Docstrings and comments-as-strings are prose; subscript keys and call arguments are handled above.
            if not (isinstance(parent, ast.Subscript) and parent.slice is node) and not isinstance(parent, (ast.Call, ast.Expr)):
                found.append(f'line {node.lineno}: kill command string {node.value!r}')

    # Helper calls: literal reasons of the matching target; helpers and forward are never aliased.
    server_sites = []
    for node in ast.walk(module):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in (*HELPERS, 'forward'):
            parent = parents.get(id(node))
            if node.id == 'forward':
                ok = (isinstance(parent, ast.Call) and dotted(parent) == 'signal.signal' and node in parent.args[1:])
            else:
                ok = isinstance(parent, ast.Call) and parent.func is node
            if not ok:
                found.append(f'line {node.lineno}: {node.id} used other than ' + ('installed by signal.signal' if node.id == 'forward' else 'as a direct call'))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in HELPERS:
            reason = node.args[0] if node.args else None
            if not (isinstance(reason, ast.Constant) and isinstance(reason.value, str)):
                found.append(f'line {node.lineno}: {node.func.id} reason is not a string literal')
                continue
            if SIGNAL_SITES.get(reason.value) != HELPERS[node.func.id]:
                found.append(f'line {node.lineno}: {reason.value!r} is not a {HELPERS[node.func.id]} reason')
            if node.func.id == '_signal_server':
                server_sites.append((*site(node, parents), reason.value))
    if sorted(server_sites) != SERVER_SITES:
        found.append(f'server signal sites changed: {sorted(server_sites)}')

    # Helper bodies: the server helper never names the worker and the worker helper never the server.
    for name, forbidden in (('_signal_worker', SERVER_NAMES), ('_signal_server', WORKER_NAMES)):
        names = {n.id for n in ast.walk(defs[name][0]) if isinstance(n, ast.Name)}
        if names & forbidden:
            found.append(f'{name} names {sorted(names & forbidden)}')
    for node in ast.walk(defs['_signal_server'][0]):
        if isinstance(node, ast.Call) and called(node) in SIGNAL_NAMES:
            target = node.args[0] if called(node) in GROUP_HELPERS and node.args else getattr(node.func, 'value', None)
            if not (isinstance(target, ast.Name) and target.id == 'child'):
                found.append(f'line {node.lineno}: _signal_server signals {ast.unparse(target) if target else "?"}, not child')
    for node in ast.walk(defs['_signal_worker'][0]):
        if isinstance(node, ast.Call) and called(node) in SIGNAL_NAMES:
            if called(node) not in GROUP_HELPERS or not node.args or ast.unparse(node.args[0]) != "w['proc']":
                found.append(f'line {node.lineno}: _signal_worker signals other than the worker group')

    # The worker record holds only the worker Popen; its identities grow only from its own tree sample.
    for node in ast.walk(module):
        function = site(node, parents)[0] if hasattr(node, 'lineno') else None
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(node, ast.AugAssign) and isinstance(target, ast.Name) and target.id in ('worker', 'proc', 'child'):
                    found.append(f'line {node.lineno}: augmented assignment to {target.id}')
                if isinstance(target, ast.Name) and target.id == 'worker' and not (isinstance(value, ast.Constant) and value.value is None):
                    ok = (function == 'start_worker' and isinstance(value, ast.Call) and dotted(value) == 'dict'
                          and {k.arg: ast.unparse(k.value) for k in value.keywords}.get('proc') == 'proc'
                          and {k.arg: ast.unparse(k.value) for k in value.keywords}.get('known') in ('{}', '{proc.pid: born}'))
                    if not ok:
                        found.append(f'line {node.lineno}: worker assigned something other than the worker Popen')
                if isinstance(target, ast.Name) and target.id == 'proc' and function == 'start_worker':
                    if not (isinstance(value, ast.Call) and dotted(value) == 'subprocess.Popen'
                            and isinstance(value.args[0], ast.Name) and value.args[0].id == 'argv'):
                        found.append(f'line {node.lineno}: the worker process is not the Popen of the worker command')
                if isinstance(target, ast.Name) and target.id == 'child' and not (isinstance(value, ast.Constant) and value.value is None):
                    if not (isinstance(value, ast.Call) and dotted(value) == 'subprocess.Popen'
                            and isinstance(value.args[0], ast.Name) and value.args[0].id == 'command'):
                        found.append(f'line {node.lineno}: child assigned something other than the server Popen')
                for t in ast.walk(target):
                    if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and t.slice.value in ('proc', 'pid', *IDENTITY_KEYS):
                        found.append(f'line {node.lineno}: the worker record\'s {t.slice.value} is reassigned')
        # Any method call on a record's known/identities set (update, add, |=-like, ...) except the pinned one.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            if (isinstance(receiver, ast.Subscript) and isinstance(receiver.slice, ast.Constant) and receiver.slice.value in IDENTITY_KEYS
                    and not (function == 'worker_tree' and ast.unparse(node) == "w['known'].update(tree['identities'])")):
                found.append(f'line {node.lineno}: the worker identity set is changed other than from its own sample')

    # Process modules only in the pinned import forms; no aliased import beyond the pinned one.
    process_imports = []
    for node in ast.walk(module):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            function = site(node, parents)[0]
            if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or '').split('.')[0] in PROCESS_MODULES:
                found.append(f'line {node.lineno}: from-import of {node.module}')
            for alias in node.names:
                if alias.asname and (alias.name, alias.asname, function) not in IMPORT_ALIASES:
                    found.append(f'line {node.lineno}: aliased import of {alias.name} as {alias.asname}')
                if isinstance(node, ast.Import) and alias.name.split('.')[0] in PROCESS_MODULES:
                    process_imports.append((alias.name, function))
    if sorted(process_imports) != PROCESS_IMPORTS:
        found.append(f'process module imports changed: {sorted(process_imports)}')
    return found+control_flow(module, parents, supervise)+bindings(module, parents, supervise, inside['_signal_server'])


def _bound(node):
    """The name ``node`` binds when it is not an ``ast.Name`` (argument, def, except-as, import, match)."""
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
        return node.name
    if isinstance(node, ast.alias):
        return node.asname or node.name.split('.')[0]
    return None


def bindings(module, parents, supervise, server_helper):
    """Reflection; the server Popen's and the saved handlers' bindings and reads; exits past ``except Exception``."""
    if supervise is None:
        return []
    found, roots, uses, exits = [], [], [], []
    in_supervise = {id(n) for n in ast.walk(supervise)}
    for node in ast.walk(module):
        where = f'line {getattr(node, "lineno", "?")}'
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        parent = parents.get(id(node))
        # Reflection: handlers, closure cells, frames, builtins and object state by another route.
        if name in REFLECTION or (name and name.startswith('__') and name.endswith('__') and name != '__name__'):
            found.append(f'{where}: {name} can reach a handler, closure or object state by reflection')
        if isinstance(node, ast.Attribute) and ast.unparse(node) == 'sys.modules':
            found.append(f'{where}: sys.modules can import a module by a computed name')
        if name in EXIT_CALLS and id(node) not in in_supervise:   # inside supervise: control_flow
            found.append(f'{where}: {name} can end supervision or signal the process')
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
            if any(m.split('.')[0] in BANNED_MODULES for m in modules):
                found.append(f'{where}: import of {", ".join(modules)}')
            if any(a.name in REFLECTION for a in node.names):
                found.append(f'{where}: import of {", ".join(a.name for a in node.names)}')
        if isinstance(node, ast.Call) and called(node) in ('getattr', 'hasattr') and len(node.args) > 1:
            attr = node.args[1].value if isinstance(node.args[1], ast.Constant) else None
            if isinstance(attr, str) and (attr in SIGNAL_NAMES | EXIT_CALLS | EXIT_EXCEPTIONS | REFLECTION | POPEN_STATE
                                          or attr.startswith('__')):
                found.append(f'{where}: getattr names {attr!r}')
        # Exceptions that escape except Exception, under any spelling, only at their pinned sites.
        text = node.value.strip() if isinstance(node, ast.Constant) and isinstance(node.value, str) else None
        if name in EXIT_EXCEPTIONS or text in EXIT_EXCEPTIONS:
            exits.append((name or text, *site(node, parents)))
        # Popen state is never written, and nothing inside supervise stores an attribute at all.
        if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
            if node.attr in POPEN_STATE or id(node) in in_supervise:
                found.append(f'{where}: stores {ast.unparse(node)}')
        # child and handlers are bound only at their pinned sites, under any binding form.
        bound = node.id if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load) else _bound(node)
        if bound in ('child', 'handlers'):
            roots.append((bound, *site(node, parents)))
        if isinstance(node, (ast.Nonlocal, ast.Global)) and set(node.names) & {'child', 'handlers'}:
            found.append(f'{where}: {type(node).__name__.lower()} {sorted(set(node.names) & {"child", "handlers"})}')
        # child is only read for its pid/poll/returncode/send_signal, compared with None, or stopped by the server helper.
        if isinstance(node, ast.Name) and node.id == 'child' and isinstance(node.ctx, ast.Load):
            ok = ((isinstance(parent, ast.Attribute) and parent.value is node and parent.attr in CHILD_ATTRIBUTES)
                  or (isinstance(parent, ast.Compare) and len(parent.ops) == 1 and isinstance(parent.ops[0], (ast.Is, ast.IsNot))
                      and isinstance(parent.comparators[0], ast.Constant) and parent.comparators[0].value is None)
                  or (isinstance(parent, ast.Call) and called(parent) == 'stop_group' and parent.args[:1] == [node]
                      and id(node) in server_helper))
            if not ok:
                found.append(f'{where}: child used other than for its pid, poll, returncode or a None check')
        # handlers is read only to install and to restore; never called through.
        if isinstance(node, ast.Name) and node.id == 'handlers' and isinstance(node.ctx, ast.Load):
            grand = parents.get(id(parent))
            function, path = site(node, parents)
            role = 'other'
            if isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.ctx, ast.Store):
                role = 'install'
            elif (isinstance(parent, ast.Attribute) and parent.attr == 'items' and isinstance(grand, ast.Call)
                  and grand.func is parent and not grand.args and isinstance(parents.get(id(grand)), ast.For)
                  and parents[id(grand)].iter is grand):
                role = 'restore'
            uses.append((role if function == 'supervise' else 'other', path))
        if isinstance(node, ast.Name) and node.id == 'handler' and isinstance(node.ctx, ast.Load):
            if not (isinstance(parent, ast.Call) and dotted(parent) == 'signal.signal' and node in parent.args[1:]):
                found.append(f'{where}: a saved handler used other than restored by signal.signal')
    if sorted(roots) != ROOT_SITES:
        found.append(f'child/handlers binding sites changed: {sorted(roots)}')
    if sorted(uses) != HANDLER_USES:
        found.append(f'handlers read sites changed: {sorted(uses)}')
    if sorted(exits) != EXIT_EXCEPTION_SITES:
        found.append(f'exit exception references changed: {sorted(exits)}')
    return found


def _broad(handler):
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return handler.type is None or any(isinstance(t, ast.Name) and t.id in BROAD for t in types)


def _contained(node, parents):
    """True when ``node`` sits in the body of a broad ``try`` inside a supervision loop."""
    child, parent = node, parents.get(id(node))
    while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if isinstance(parent, ast.While) and ast.unparse(parent.test) == 'child.poll() is None':
            return False
        if isinstance(parent, ast.Try) and _field(parent, child) == 'body' and any(_broad(h) for h in parent.handlers):
            if any(isinstance(p, ast.While) and ast.unparse(p.test) == 'child.poll() is None' for p in _ancestors(parent, parents)):
                return True
        child, parent = parent, parents.get(id(parent))
    return False


def _ancestors(node, parents):
    parent = parents.get(id(node))
    while parent is not None and not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
        yield parent
        parent = parents.get(id(parent))


def control_flow(module, parents, supervise):
    """Shutdown state stores, process exits and ways out of the loops in ``supervise``."""
    if supervise is None:
        return ['supervise is not defined']
    found, states, exits, installs = [], [], [], []
    for node in ast.walk(module):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name.rsplit('.', 1)[-1] in EXIT_CALLS | EXIT_EXCEPTIONS:
                    found.append(f'line {node.lineno}: import of {alias.name}')
    for node in ast.walk(supervise):
        where = f'line {getattr(node, "lineno", "?")}'
        if isinstance(node, ast.Name) and node.id in ('stopping', 'forced') and not isinstance(node.ctx, ast.Load):
            states.append((node.id, *site(node, parents)))
        if isinstance(node, (ast.Nonlocal, ast.Global)):
            for name in set(node.names) & {'stopping', 'forced'}:
                if isinstance(node, ast.Global) or site(node, parents)[0] != 'forward':
                    found.append(f'{where}: {type(node).__name__.lower()} {name} outside forward')
        if isinstance(node, ast.Attribute) and node.attr in EXIT_CALLS:
            found.append(f'{where}: .{node.attr} can end supervision or signal the process')
        if isinstance(node, ast.Name) and node.id in EXIT_CALLS:
            found.append(f'{where}: {node.id} can end supervision or signal the process')
        if isinstance(node, ast.Raise) and node.exc is not None:
            raised = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if any(isinstance(n, ast.Name) and n.id in EXIT_EXCEPTIONS for n in ast.walk(raised)):
                found.append(f'{where}: raises {ast.unparse(node.exc)}')
        if isinstance(node, ast.Call) and dotted(node) == 'signal.signal':
            function, path = site(node, parents)
            if function != 'supervise' or path not in SIGNAL_INSTALLS:
                found.append(f'{where}: signal.signal outside the pinned install/restore sites')
            installs.append(path)
        if isinstance(node, (ast.Break, ast.Return, ast.Raise)):
            function, path = site(node, parents)
            if function == 'supervise' and not (isinstance(node, ast.Raise) and _contained(node, parents)):
                exits.append((type(node).__name__.lower(), path))
    if sorted(states) != STATE_SITES:
        found.append(f'stopping/forced store sites changed: {sorted(states)}')
    if sorted(exits) != EXIT_SITES:
        found.append(f'supervise exit sites changed: {sorted(exits)}')
    if sorted(installs) != SIGNAL_INSTALLS:
        found.append(f'signal.signal sites changed: {sorted(installs)}')
    return found


def real():
    return SOURCE.read_text()


# ------------------------------------------------------------------ the module as it is
def test_the_supervisor_has_no_signal_path_outside_the_reviewed_sites():
    assert violations(real()) == []


def test_every_listed_reason_is_a_real_call_site_of_its_target():
    module = ast.parse(real())
    used = {}
    for node in ast.walk(module):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in HELPERS:
            used.setdefault(node.args[0].value, set()).add(HELPERS[node.func.id])
    assert used == {reason: {target} for reason, target in SIGNAL_SITES.items()}
    raw = {called(n) for n in ast.walk(module) if isinstance(n, ast.Call) and called(n) in SIGNAL_NAMES}
    assert raw == {'send_signal', 'stop_group', 'kill_group'}


def test_the_server_is_signalled_only_for_shutdown_and_exited_cleanup():
    assert {r for r, target in SIGNAL_SITES.items() if target == 'server'} == SERVER_REASONS
    assert {r for r, target in SIGNAL_SITES.items() if target == 'worker'} == WORKER_REASONS
    assert set(SIGNAL_SITES.values()) == {'server', 'worker'}
    assert len(SERVER_SITES) == 5 and {reason for _, _, reason in SERVER_SITES} == SERVER_REASONS


def test_the_pr_a_in_process_server_stop_is_gone():
    # B5: no memory or model fault stops the server; these names must not come back.
    gone = ('temporary_stop', 'temporary_server_stop', 'unresponsive_model', 'pr_a_in_process')
    for path in [*(APP/'wsp').rglob('*.py'), *(APP/'web').rglob('*.js')]:
        text = path.read_text()
        assert not [g for g in gone if g in text], path


def memory_guard_group_calls(path):
    """(function, line) of every reference to memory_guard's stop_group/kill_group in ``path``.

    Follows the binding: direct names inside memory_guard, names imported from it (under any
    alias) and attributes of the module under any alias. Same-named methods of other classes
    (``agent_auth``'s and ``native_harness``'s ``self.kill_group``/``self._kill_group``, which
    stop those classes' own subprocesses) are different functions and are not counted.
    """
    module = ast.parse(path.read_text())
    parents = _parents(module)
    names, modules = (set(GROUP_HELPERS) if path.name == 'memory_guard.py' else set()), set()
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and (node.module or '').endswith('memory_guard'):
            names |= {a.asname or a.name for a in node.names if a.name in GROUP_HELPERS}
            if any(a.name == '*' for a in node.names):names |= GROUP_HELPERS
        if isinstance(node, ast.ImportFrom) and not (node.module or '').endswith('memory_guard'):
            modules |= {a.asname or a.name for a in node.names if a.name == 'memory_guard'}
        if isinstance(node, ast.Import):
            modules |= {a.asname or a.name for a in node.names if a.name.endswith('memory_guard')}
    found = []
    for node in ast.walk(module):
        hit = ((isinstance(node, ast.Name) and node.id in names and isinstance(node.ctx, ast.Load))
               or (isinstance(node, ast.Attribute) and node.attr in GROUP_HELPERS
                   and ast.unparse(node.value) in modules))
        if hit:
            found.append((site(node, parents)[0], node.lineno))
    return found


def test_group_stop_helpers_are_reachable_only_from_the_two_helpers():
    allowed = {
        ('runtime_supervisor.py', '_signal_server'), ('runtime_supervisor.py', '_signal_worker'),
        # The standalone outer acceptance guard (python -m wsp.memory_guard): a separate program
        # that owns its own child; it may keep its use.
        ('memory_guard.py', 'supervise'),
        # The IDE server's own teardown of its agent-provider processes; it runs in the server,
        # not the supervisor, and never sees the server or worker identity.
        ('lifecycle.py', 'terminate'),
    }
    seen = set()
    for path in sorted((APP/'wsp').rglob('*.py')):
        for function, line in memory_guard_group_calls(path):
            assert (path.name, function) in allowed, (str(path.relative_to(APP)), line, function)
            seen.add((path.name, function))
        defined = {n.name for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
        assert not (defined & GROUP_HELPERS) or path.name == 'memory_guard.py', path
    assert seen == allowed


def test_the_cross_module_check_follows_aliases(tmp_path):
    for text in ("from .memory_guard import stop_group as sg\ndef run(c,k):\n    sg(c,k)\n",
                 "from . import memory_guard as mg\ndef run(c,k):\n    mg.kill_group(c,k)\n",
                 "from .memory_guard import *\ndef run(c,k):\n    stop_group(c,k)\n"):
        path = tmp_path/'other.py'
        path.write_text(text)
        assert memory_guard_group_calls(path) == [('run', 3)], text


# ------------------------------------------------------------------ the checker rejects each bypass
MEMORY_PATH = "        report_lost();ladder.reset()\n    def poll_worker(now):"
WORKER_HELPER = "        w=worker\n"


KILL_BRANCH = "                            kill_worker(sample);previous=None\n"


def mutate(snippet, anchor=MEMORY_PATH, after=True, indent=8):
    source = real()
    assert source.count(anchor) == 1, anchor
    indented = ''.join(' '*indent+line+'\n' for line in snippet.splitlines())
    head, tail = (anchor.split('\n', 1)[0]+'\n', anchor.split('\n', 1)[1]) if after else ('', anchor)
    return source.replace(anchor, head+indented+tail if after else indented+tail)


BYPASSES = {
    'getattr_kill': "getattr(child,'kill')()",
    'getattr_dynamic': "name='ki'+'ll'\ngetattr(os,name)(child.pid,9)",
    'methodcaller': "import operator\noperator.methodcaller('terminate')(child)",
    'alias_import_stop_group': "from .memory_guard import stop_group as sg\nsg(child,known)",
    'module_alias_stop_group': "from . import memory_guard as mg\nmg.stop_group(child,known)",
    'star_import': "from .memory_guard import *",
    'from_os_import_kill': "from os import kill\nkill(child.pid,9)",
    'os_kill': "os.kill(child.pid,signal.SIGKILL)",
    'os_killpg': "os.killpg(child.pid,signal.SIGKILL)",
    'name_alias_of_group_helper': "stopper=stop_group\nstopper(child,known)",
    'subprocess_run_kill': "subprocess.run(['kill','-9',str(child.pid)])",
    'subprocess_variable_command': "cmd=['kill',str(child.pid)]\nsubprocess.call(cmd)",
    'os_system_kill': "os.system('kill %d'%child.pid)",
    'os_popen_pkill': "os.popen('pkill -f wsp')",
    'psutil_kill': "psutil.Process(child.pid).kill()",
    'psutil_terminate': "psutil.Process(child.pid).terminate()",
    'psutil_send_signal': "psutil.Process(child.pid).send_signal(signal.SIGTERM)",
    'send_signal_on_child': "child.send_signal(signal.SIGTERM)",
    'listed_server_reason_in_memory_path': "_signal_server('user_shutdown',signal.SIGTERM)",
    'cleanup_reason_in_memory_path': "_signal_server('server_exited_cleanup')",
    'worker_reason_on_server_helper': "_signal_server('kill_worker',signal.SIGKILL)",
    'computed_reason': "why='user_'+'shutdown'\n_signal_server(why,signal.SIGTERM)",
    'aliased_server_helper': "stop=_signal_server\nstop('user_shutdown',signal.SIGTERM)",
    'forward_called_from_memory_path': "forward(signal.SIGTERM,None)",
    'worker_record_pointed_at_server': "worker['proc']=child",
    'worker_record_replaced': "worker=dict(proc=child,pid=child.pid,known=dict(known),owned=True,killed=False,stop=None)",
    'server_identities_into_worker_set': "worker['known'].update(known)",
    'exec_hidden_call': "exec('child.kill()')",
    'dunder_import': "__import__('os').kill(child.pid,9)",
}


@pytest.mark.parametrize('name', sorted(BYPASSES))
def test_each_bypass_in_the_memory_path_is_rejected(name):
    assert violations(mutate(BYPASSES[name])), name


@pytest.mark.parametrize('snippet', [
    "child.send_signal(sig)",                        # the worker helper signalling the server
    "stop_group(child,known,errors)",
    "kill_group(child,known,errors,sig)",
    "w={'proc':child,'known':known}",
])
def test_the_worker_helper_cannot_reach_the_server(snippet):
    source = mutate(snippet, anchor=WORKER_HELPER+"        if sig is None:remaining=stop_group(w['proc'],w['known'],errors)")
    assert violations(source), snippet


def test_the_server_helper_cannot_reach_the_worker():
    source = mutate("kill_group(worker['proc'],worker['known'],errors)", indent=12,
                    anchor="            if sig is None:return stop_group(child,known,errors)\n            if child is not None")
    assert violations(source)


def test_a_moved_or_extra_server_site_is_rejected():
    source = real()
    # The same listed reason at a second place in the teardown is a new site.
    anchor = "        if socket_dir is not None:shutil.rmtree(socket_dir,ignore_errors=True)\n"
    assert source.count(anchor) == 1
    assert any('server signal sites changed' in v for v in
               violations(source.replace(anchor, "        remaining+=_signal_server('server_exited_cleanup',errors=errors)\n"+anchor)))
    # Dropping the forced-shutdown guard moves the call.
    guarded = "                if forced:remaining+=_signal_server('shutdown_forced',errors=errors)"
    assert source.count(guarded) == 1
    assert violations(source.replace(guarded, "                if True:remaining+=_signal_server('shutdown_forced',errors=errors)"))


def test_the_checker_accepts_harmless_mentions():
    # A clean edit outside the signal paths stays clean (no false positives from the rules above).
    assert violations(mutate("evidence(lambda:record(dict(event='note',kill=1,stage='kill_worker')))")) == []


# ------------------------------------------------------------------ the checker rejects each way out of supervision
EXITS = {
    'stopping_set_in_kill_branch': "stopping=time.monotonic()",                    # reviewer case
    'forced_break_in_kill_branch': "forced=True;break",                             # reviewer case
    'raise_signal_in_kill_branch': "signal.raise_signal(signal.SIGTERM)",           # reviewer case
    'aliased_raise_signal': "from signal import raise_signal as rs\nrs(signal.SIGTERM)",
    'module_alias_raise_signal': "import signal as s\ns.raise_signal(15)",
    'os_kill_self': "os.kill(os.getpid(),signal.SIGTERM)",
    'os_killpg_self': "os.killpg(0,signal.SIGTERM)",
    'pthread_kill_self': "signal.pthread_kill(threading.main_thread().ident,signal.SIGTERM)",
    'alarm': "signal.alarm(1)",
    'setitimer': "signal.setitimer(signal.ITIMER_REAL,1)",
    'install_handler': "signal.signal(signal.SIGALRM,forward)",
    'sys_exit': "sys.exit(1)",
    'os_exit': "os._exit(1)",
    'os_abort': "os.abort()",
    'builtin_exit': "exit()",
    'builtin_quit': "quit(1)",
    'raise_system_exit': "raise SystemExit(1)",
    'raise_keyboard_interrupt': "raise KeyboardInterrupt",
    'raise_base_exception': "raise BaseException('stop')",
    'break_out_of_the_loop': "break",
    'return_from_supervise': "return 1",
    'nonlocal_store': "def later():\n    nonlocal stopping\n    stopping=0\nlater()",
    'walrus_store': "(forced:=True)",
    'augmented_store': "forced|=True",
}


@pytest.mark.parametrize('name', sorted(EXITS))
def test_each_way_out_of_supervision_in_the_kill_branch_is_rejected(name):
    assert violations(mutate(EXITS[name], anchor=KILL_BRANCH, indent=28)), name


def test_the_reviewer_cases_are_rejected_for_the_right_reason():
    def found(snippet):
        return ' | '.join(violations(mutate(snippet, anchor=KILL_BRANCH, indent=28)))
    assert 'stopping/forced store sites changed' in found("stopping=time.monotonic()")
    both = found("forced=True;break")
    assert 'stopping/forced store sites changed' in both and 'supervise exit sites changed' in both
    assert '.raise_signal can end supervision' in found("signal.raise_signal(signal.SIGTERM)")


# Second review: each passed the checker before, inserted after kill_worker(sample);previous=None.
# (snippet, the reason it must be rejected for)
BINDING_BYPASSES = {
    'fake_returncode': ("child.returncode=0", 'stores child.returncode'),     # poll() then 0: teardown stops the server
    'getsignal_forward': ("signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)", 'getsignal can reach'),
    'psutil_suspend': ("psutil.Process(child.pid).suspend()", '.suspend outside the signal helpers'),
    'psutil_resume': ("psutil.Process(child.pid).resume()", '.resume outside the signal helpers'),
    'getattr_system_exit': ("raise getattr(__builtins__,'SystemExit')(0)", "getattr names 'SystemExit'"),
    'aliased_keyboard_interrupt': ("E=KeyboardInterrupt\nraise E()", 'exit exception references changed'),
    'child_rebound_to_none': ("child=None", 'child/handlers binding sites changed'),
    'handlers_called': ("handlers[signal.SIGINT](signal.SIGINT,None)", 'handlers read sites changed'),
    # Neighbours of the same holes.
    'setattr_child': ("setattr(child,'returncode',0)", 'setattr can reach'),
    'child_aliased': ("server=child\nserver.returncode=0", 'child used other than'),
    'child_passed_away': ("print(child)", 'child used other than'),
    'handlers_get_called': ("handlers.get(signal.SIGTERM)(signal.SIGTERM,None)", 'handlers read sites changed'),
    'handlers_rebound': ("handlers={signal.SIGTERM:forward}", 'child/handlers binding sites changed'),
    'default_int_handler': ("signal.default_int_handler(signal.SIGINT,None)", 'default_int_handler can reach'),
    'getattr_sys_exit': ("getattr(sys,'exit')(0)", "getattr names 'exit'"),
    'builtins_module': ("import builtins\nraise builtins.SystemExit(0)", 'import of builtins'),
    'frame_locals': ("sys._getframe().f_locals['forward'](signal.SIGTERM,None)", '_getframe can reach'),
    'closure_cell': ("kill_worker.__closure__[0].cell_contents", '__closure__ can reach'),
    'sys_modules': ("sys.modules['ctypes']", 'sys.modules can import'),
    'string_exception_name': ("raise vars(__builtins__)['GeneratorExit']", 'exit exception references changed'),
    'cancelled_error': ("import asyncio\nraise asyncio.CancelledError()", 'exit exception references changed'),
    'except_as_child': ("try:pass\nexcept Exception as child:pass", 'child/handlers binding sites changed'),
    'pidfd_send_signal': ("signal.pidfd_send_signal(0,signal.SIGTERM)", '.pidfd_send_signal outside'),
}


@pytest.mark.parametrize('name', sorted(BINDING_BYPASSES))
def test_each_binding_or_reflection_bypass_in_the_kill_branch_is_rejected(name):
    snippet, reason = BINDING_BYPASSES[name]
    found = violations(mutate(snippet, anchor=KILL_BRANCH, indent=28))
    assert any(reason in v for v in found), (name, found)


WORKER_TREE_UPDATE = "        w['known'].update(tree['identities'])\n"
# Third review: each passed before. (snippet, anchor, reason); inserted before the anchor line.
IDENTITY_BYPASSES = {
    'augmented_worker_known': ("worker['known']|=known", KILL_BRANCH, 28, "the worker record's known is reassigned"),
    'augmented_w_identities': ("tree['identities']|=known", WORKER_TREE_UPDATE, 8, "the worker record's identities is reassigned"),
    'tree_identities_update': ("tree['identities'].update(known)", WORKER_TREE_UPDATE, 8, 'the worker identity set is changed'),
    'worker_known_add': ("worker['known'].setdefault(child.pid,0)", KILL_BRANCH, 28, 'the worker identity set is changed'),
    'identities_item_store': ("tree['identities'][child.pid]=0", WORKER_TREE_UPDATE, 8, "the worker record's identities is reassigned"),
    'worker_augmented': ("worker|={'proc':child}", KILL_BRANCH, 28, 'augmented assignment to worker'),
}


@pytest.mark.parametrize('name', sorted(IDENTITY_BYPASSES))
def test_each_change_to_a_worker_identity_set_is_rejected(name):
    snippet, anchor, indent, reason = IDENTITY_BYPASSES[name]
    found = violations(mutate(snippet, anchor=anchor, after=False, indent=indent))
    assert any(reason in v for v in found), (name, found)


ALIASED_IMPORTS = {
    'import_os_as': ("import os as o\no.system('true')", 'aliased import of os as o'),
    'from_os_import_system_as': ("from os import system as run_shell", 'from-import of os'),
    'from_os_import_posix_spawn_as': ("from os import posix_spawn as ps", 'aliased import of posix_spawn as ps'),
    'from_subprocess_import': ("from subprocess import Popen", 'from-import of subprocess'),
    'from_psutil_import': ("from psutil import Process", 'from-import of psutil'),
    'other_module_aliased': ("import json as j", 'aliased import of json as j'),
    'second_plain_os_import': ("import os", 'process module imports changed'),
    'plain_asyncio_import': ("import asyncio", 'process module imports changed'),
}


@pytest.mark.parametrize('name', sorted(ALIASED_IMPORTS))
def test_aliased_and_from_imports_of_process_modules_are_rejected(name):
    snippet, reason = ALIASED_IMPORTS[name]
    found = violations(mutate(snippet, anchor=KILL_BRANCH, indent=28))
    assert any(reason in v for v in found), (name, found)


def test_the_real_import_forms_are_pinned():
    module = ast.parse(real())
    parents = _parents(module)
    plain = sorted((a.name, site(n, parents)[0]) for n in ast.walk(module) if isinstance(n, ast.Import)
                   for a in n.names if a.name.split('.')[0] in PROCESS_MODULES)
    aliased = [(a.name, a.asname, site(n, parents)[0]) for n in ast.walk(module) if isinstance(n, (ast.Import, ast.ImportFrom))
               for a in n.names if a.asname]
    assert plain == PROCESS_IMPORTS and aliased == IMPORT_ALIASES


def test_an_exit_raised_by_a_module_level_helper_is_rejected():
    # A helper outside supervise that ends supervision, called from the kill branch.
    source = mutate("_stop_now()", anchor=KILL_BRANCH, indent=28)
    for helper, reason in (("def _stop_now():\n    raise SystemExit(0)\n", 'exit exception references changed'),
                           ("def _stop_now():\n    signal.raise_signal(signal.SIGTERM)\n", 'raise_signal can end supervision'),
                           ("def _stop_now():\n    sys.exit(0)\n", 'exit can end supervision'),
                           ("def _stop_now(c=None):\n    c.returncode=0\n", 'stores c.returncode')):
        found = violations(source.replace('\ndef supervise(', '\n'+helper+'\n\ndef supervise(', 1))
        assert any(reason in v for v in found), (helper, found)


@pytest.mark.parametrize('anchor,indent', [
    ("            if stopping is not None and now-stopping>30:forced=True;break\n", 12),   # the fault loop
    ("                    previous=None\n                    evidence(lambda:lease(False,lost or latest)", 20),  # sampling except
    ("            try:request=json.loads((directory/'request.json').read_text())\n", 12),  # main loop, unprotected
])
def test_a_raise_out_of_a_supervision_loop_is_rejected(anchor, indent):
    found = violations(mutate("raise RuntimeError('stop')", anchor=anchor, after='try:' not in anchor, indent=indent))
    assert any('supervise exit sites changed' in v for v in found), found


def test_a_contained_raise_in_the_request_handler_stays_allowed():
    # A refusal raised in the request try is answered by its except Exception, not a way out.
    anchor = "                    if action=='release':\n"
    assert violations(mutate("raise ValueError('refused')", anchor=anchor, after=False, indent=20)) == []
