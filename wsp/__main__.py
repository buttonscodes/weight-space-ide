"""Command line.

    python -m wsp serve [--model facts] [--port 8765] [--root workspaces]     the IDE
    python -m wsp open  <hub repo id | local directory> [--pull]              a workspace over any safetensors checkpoint
    python -m wsp pull  <workspace>                                           download a Hub workspace's weights (to run it)
    python -m wsp export --model <workspace> <directory>                      write the edited checkpoint
    python -m wsp build <model>                                               build one of the bundled models
    python -m wsp models                                                      list models
    python -m wsp brief  --model facts                                        print the programmer's brief
    python -m wsp plan   --model facts [--budget 40000] [--format auto]       print a chunk plan
    python -m wsp apply  --model blank-reverse examples/reverse.wscript       apply a weight script
    python -m wsp test   --model facts                                        run the task's test suite
    python -m wsp mcp    [--url http://127.0.0.1:8765]                        stdio MCP server (for harnesses)
    python -m wsp worker --runtime-dir D --socket S --workspace W              model worker (started by the supervisor)

--model accepts a workspace name, a bundled model name, a Hub repo id or a local checkpoint directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="wsp", description="weight-space programming IDE")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--root", default=os.environ.get("WSP_ROOT", "workspaces"))
        p.add_argument("--model", default="facts")
        return p

    from .runtime_config import DEFAULTS, INTS, MEMORY
    for command in ('serve','doctor'):
        s = sub.add_parser(command, help='serve the IDE' if command == 'serve' else 'show effective config/setup without loading weights')
        s.add_argument('--config', help='explicit version-1 JSON config; default: per-user platform/XDG config')
        s.add_argument('--profile', help='named user profile (does not authorize agent model loads)')
        for key in DEFAULTS:
            if key == 'model_roots':
                s.add_argument('--model-root', dest=key, action='append')
            else:
                s.add_argument('--'+key.replace('_','-'), type=int if key in INTS else float if key in MEMORY else str)
        if command == 'doctor': s.add_argument('--inspect-model',action='store_true',help='read GGUF header/directory only for parameter count and memory estimate; no payload scan')
        if command == 'serve':
            s.add_argument('--empty',action='store_true',help='start without opening/building any workspace')
            s.add_argument('--runtime-child',help=argparse.SUPPRESS)
    b = sub.add_parser("build")
    b.add_argument("name")
    b.add_argument("--root", default=os.environ.get("WSP_ROOT", "workspaces"))
    sub.add_parser("models").add_argument("--root", default=os.environ.get("WSP_ROOT", "workspaces"))
    o = sub.add_parser("open")
    o.add_argument("spec")
    o.add_argument("--root", default=os.environ.get("WSP_ROOT", "workspaces"))
    o.add_argument("--pull", action="store_true")
    pl = sub.add_parser("pull")
    pl.add_argument("name")
    pl.add_argument("--root", default=os.environ.get("WSP_ROOT", "workspaces"))
    common(sub.add_parser("export")).add_argument("directory")
    common(sub.add_parser("brief")).add_argument("--mode", default="tools")
    p = common(sub.add_parser("plan"))
    p.add_argument("--budget", type=int, default=40_000)
    p.add_argument("--format", default="auto")
    p.add_argument("selection", nargs="*")
    a = common(sub.add_parser("apply"))
    a.add_argument("script")
    # The Largest weight patch setting (wsp.edit_limit), read from the same configuration as serve.
    a.add_argument("--config", help="explicit version-1 JSON config; default: per-user platform/XDG config")
    a.add_argument("--profile", help="named user profile")
    a.add_argument("--patch-limit-gib", dest="patch_limit_gib", type=float,
                   help="largest patch this edit may write, GiB of FP32 edits (default: the saved setting, else Auto)")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("-m", "--message", default="")
    common(sub.add_parser("test")).add_argument("--suite", default="")
    at = common(sub.add_parser("atlas", help="build the atlas of a model, or print it"))
    at.add_argument("--stages", default="", help="comma-separated: weights,wiring,observed (default: weights, wiring when cheap, observed when "
                         "the model runs in this process; a model in the model worker builds observed only when named)")
    at.add_argument("--fetch", action="store_true", help="download the tensors a remote model needs")
    at.add_argument("--corpus", default="", help="probe text for the observed stage (default: workspace corpus.txt, else bundled)")
    at.add_argument("--tokens", type=int, default=0)
    at.add_argument("--force", action="store_true")
    at.add_argument("--show", default="", help="print instead of building: overview | L<layer>N<n> | concepts")
    m = sub.add_parser("mcp")
    m.add_argument("--url", default=os.environ.get("WSP_URL", "http://127.0.0.1:8765"))
    w = sub.add_parser("worker", help="internal: the model worker started by the runtime supervisor (token in WSP_WORKER_TOKEN)")
    w.add_argument("--runtime-dir", required=True)
    w.add_argument("--socket", required=True)
    w.add_argument("--workspace", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "worker":
        # Lazy: the worker publishes idle GPU telemetry before torch is imported.
        from .worker import main as worker_main
        raise SystemExit(worker_main(args.runtime_dir, args.socket, args.workspace))

    if args.cmd in ('serve','doctor'):
        from .runtime_config import resolve, activate, doctor
        try:
            if getattr(args,'runtime_child',None):
                from pathlib import Path
                directory = Path(args.runtime_child).resolve()
                if str(directory) != os.environ.get('WSP_RUNTIME_DIR'):
                    raise ValueError('Internal server entry requires its external supervisor')
                settings = json.loads((directory/'settings.json').read_text())
            else:
                settings = resolve(args.config,args.profile,{k:getattr(args,k) for k in DEFAULTS})
            if args.cmd == 'doctor':
                print(json.dumps(doctor(settings,inspect_model=args.inspect_model),indent=2)); return
            backend = activate(settings)
            if not getattr(args,'runtime_child',None) and settings['platform'] in {'Darwin','Linux'}:
                from .runtime_supervisor import supervise
                raise SystemExit(supervise(settings,empty=args.empty))
            for key,value in settings['values'].items(): setattr(args,key,value)
        except (ValueError,OSError) as exc:
            ap.error(str(exc))
    if args.cmd == "mcp":
        from .mcp import serve_stdio
        serve_stdio(args.url)
        return
    from .session import Session
    os.makedirs(args.root, exist_ok=True)
    sess = Session(args.root)
    if args.cmd == "models":
        for w in sess.available():
            print(f"{'*' if w['built'] else ' '} {w['name']:18s} {w['description']}")
        return
    if args.cmd in ("build", "open", "pull"):
        ws = sess.open(args.name if args.cmd != "open" else args.spec, pull=args.cmd == "pull" or getattr(args, "pull", False))
        print(json.dumps(ws.summary(), indent=1))
        return
    if args.cmd == 'serve':
        sess.settings = settings
        if args.runtime_child:
            from .runtime_supervisor import RuntimeClient
            # Supervised: GGUF runners run in the supervisor-spawned model worker.
            sess.runtime = RuntimeClient(args.runtime_child,settings,worker=True)
        sess.model_roots = settings['values']['model_roots']
    if not (args.cmd == 'serve' and args.empty): sess.open(args.model)
    if args.cmd == "serve":
        from .lifecycle import serve
        from .server import create_app
        url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
        print(f"weight-space programming IDE: {url}   (workspace: {sess.ws.dir if sess.ws else 'none'})")
        serve(create_app(sess, url), host=args.host, port=args.port, log_level="warning")
    elif args.cmd == "atlas":
        import re
        from . import atlas as A
        from .atlas import report
        if not args.show:
            try:
                A.build(sess.ws, stages=[x for x in args.stages.split(",") if x] or None, fetch=args.fetch,
                        corpus=args.corpus or None, tokens=args.tokens or None, force=args.force)
            except A.Busy as exc:                        # another build of this atlas runs: say so, no traceback
                print(f"wsp atlas: {exc}.", file=sys.stderr)
                raise SystemExit(1) from None
        at_ = A.Atlas.of(sess.ws)
        mm = re.fullmatch(r"L(\d+)N(\d+)", args.show or "")
        print(report.unit(at_, int(mm.group(1)), int(mm.group(2))) if mm else report.concepts(at_) if args.show == "concepts" else report.overview(at_))
    elif args.cmd == "export":
        for f in sess.ws.export(args.directory):
            print(f)
    elif args.cmd == "brief":
        print(sess.brief(args.mode))
    elif args.cmd == "plan":
        from .tools import plan_chunks
        print(plan_chunks(sess, args.selection or None, args.budget, args.format))
    elif args.cmd == "apply":
        from .tools import _report
        sess.set_edit_settings(_apply_settings(args, ap))
        print(_report(sess.apply(open(args.script).read(), args.message, dry_run=args.dry_run)))
    elif args.cmd == "test":
        from .testsuite import summarize
        r = sess.tests(args.suite or None)
        print(summarize(r) if r else "no test suite")


def _apply_settings(args, ap) -> dict:
    """The settings values `wsp apply` edits read: the Largest weight patch from the configuration (and
    --patch-limit-gib), as the server reads it.

    Explicit configuration (--config, --profile, --patch-limit-gib, WSP_CONFIG, WSP_PROFILE) that does not
    resolve is an error. Otherwise, when the settings do not resolve:
    - a default configuration that sets a Largest weight patch is not dropped: if the configuration itself is
      valid (the fault is elsewhere, such as a WSP_ environment variable), its setting applies; if not, the
      edit is refused with the reason;
    - with no readable default configuration, or one that sets no limit, the setting is Auto with a note (apply
      read no configuration before the setting existed)."""
    import sys
    from .runtime_config import DEFAULTS, resolve, user_dir
    try:
        return resolve(args.config, args.profile, {"patch_limit_gib": args.patch_limit_gib})["values"]
    except (ValueError, OSError) as exc:
        explicit = [name for name, given in (("--config", args.config), ("--profile", args.profile),
                                             ("--patch-limit-gib", args.patch_limit_gib is not None),
                                             ("WSP_CONFIG", os.environ.get("WSP_CONFIG")),
                                             ("WSP_PROFILE", os.environ.get("WSP_PROFILE"))) if given]
        if explicit:
            ap.error(f"{exc} (given by {', '.join(explicit)})")
        path = user_dir()/"config.json"
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            document = None
        defaults = document.get("defaults") if isinstance(document, dict) else None
        saved = defaults.get("patch_limit_gib") if isinstance(defaults, dict) else None
        if saved is None:
            print(f"wsp apply: settings not read ({exc}); Largest weight patch is Auto", file=sys.stderr)
            return dict(DEFAULTS)
        try:
            # The configuration alone, without the WSP_ environment: is the saved setting itself valid?
            clean = {k: v for k, v in os.environ.items() if not k.startswith("WSP_")}
            limit = resolve(str(path), env=clean, document=document)["values"]["patch_limit_gib"]
        except (ValueError, OSError) as own:
            ap.error(f"{own}. {path} sets Largest weight patch (patch_limit_gib {saved!r}), and a saved setting is "
                     "not dropped: fix the configuration, then apply again")
        print(f"wsp apply: settings not read ({exc}); the saved Largest weight patch of {limit:g} GiB from {path} "
              "still applies", file=sys.stderr)
        return dict(DEFAULTS, patch_limit_gib=limit)


if __name__ == "__main__":
    main()
