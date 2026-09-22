"""Terminal UI (Rich). `rootlogic research "topic"` plus history/usage/log inspection commands.

Ctrl-C during a run pauses at the next safe point and opens the override menu.
Press Ctrl-C twice to hard-exit.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .control import Command, Event
from .models import Plan, SubTaskDraft
from .orchestrator import Budget, Orchestrator
from .store import Store
from .backend import Backend, BackendError, parse_prices
from .filters import RULES, SourcePolicy, clean_domain

console = Console()
HOME = Path(os.environ.get("ROOTLOGIC_HOME", ".rootlogic"))

STYLE = {
    "session": "bold cyan", "memory": "magenta", "clarify": "yellow", "plan": "cyan",
    "task": "white", "source": "dim", "reflect": "blue", "analyze": "blue", "report": "green",
    "user": "yellow", "override": "bold yellow", "control": "bold yellow", "loop": "dim",
}


class TerminalUI:
    def __init__(self, auto_approve: bool = False, verbose: bool = False):
        self.auto_approve = auto_approve
        self.verbose = verbose

    # --------------------------------------------------------------- events
    def on_event(self, e: Event) -> None:
        if e.type == "source.dropped" and not self.verbose:
            return
        prefix = e.type.split(".")[0]
        style = STYLE.get(prefix, "white")
        if e.type.endswith("failed"):
            style = "bold red"
        console.print(f"[dim]{e.ts[11:19]}[/] [{style}]{e.type:<18}[/] {escape(e.message)}",
                      highlight=False)

    # --------------------------------------------------------------- questions
    def ask(self, question: str) -> str:
        return Prompt.ask(f"[bold yellow]?[/] {question} [dim](Enter to skip)[/]", default="",
                          show_default=False)

    # --------------------------------------------------------------- plan review
    def review_plan(self, plan: Plan) -> Plan | None:
        while True:
            show_plan(plan)
            if self.auto_approve:
                return plan
            choice = Prompt.ask("[bold]Plan[/] — [a]pprove, [e]dit, [q]uit", choices=["a", "e", "q"],
                                default="a")
            if choice == "a":
                return plan
            if choice == "q":
                return None
            self._edit(plan)

    def _edit(self, plan: Plan) -> None:
        console.print("[dim]Commands: drop <id> · add <question> · recency <days> · done[/]")
        while True:
            raw = Prompt.ask("edit").strip()
            cmd, _, arg = raw.partition(" ")
            if cmd == "done" or not raw:
                return
            if cmd == "drop":
                plan.subtasks = [t for t in plan.subtasks if t.id != arg]
                for t in plan.subtasks:
                    t.depends_on = [d for d in t.depends_on if d != arg]
            elif cmd == "add" and arg:
                plan.add(SubTaskDraft(question=arg, rationale="Added by user",
                                      search_queries=[arg], depends_on=[]), origin="user")
            elif cmd == "recency" and arg.isdigit():
                plan.recency_days = int(arg)
            else:
                console.print("[red]?[/]")

    # --------------------------------------------------------------- override
    def override(self, plan: Plan) -> list[Command]:
        show_plan(plan)
        console.print(Panel(
            "[bold]continue[/] · [bold]skip[/] <id> · [bold]add[/] <question> · "
            "[bold]note[/] <guidance> · [bold]stop[/] (write report now) · [bold]abort[/]",
            title="Paused — override", border_style="yellow"))
        commands: list[Command] = []
        while True:
            raw = Prompt.ask("override", default="continue").strip()
            action, _, arg = raw.partition(" ")
            if action == "continue":
                return commands
            if action in ("stop", "abort"):
                return commands + [Command(action)]  # type: ignore[arg-type]
            if action in ("skip", "add", "note") and arg:
                commands.append(Command(action, arg))  # type: ignore[arg-type]
                console.print(f"[green]queued[/] {action} {arg}")
            else:
                console.print("[red]?[/]")


def show_plan(plan: Plan) -> None:
    table = Table(title=f"Plan · {plan.objective}", title_justify="left", show_lines=False)
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("status")
    table.add_column("question")
    table.add_column("after", style="dim")
    table.add_column("origin", style="dim")
    colors = {"done": "green", "running": "yellow", "failed": "red", "skipped": "dim"}
    for t in plan.subtasks:
        c = colors.get(t.status, "white")
        table.add_row(t.id, f"[{c}]{t.status}[/]", t.question, ",".join(t.depends_on), t.origin)
    console.print(table)
    console.print(f"[dim]Recency: {plan.recency_days or 'any'} days[/]")


# =================================================================== commands


def create_engine(store: Store, ui, *, engine: str = "loop", offline: bool = False,
                  budget: Budget | None = None, home: Path = HOME,
                  backend: Backend | None = None, use_profile: bool = True,
                  source_policy: SourcePolicy | None = None):
    """Construct an engine with usage accounting wired to the store.

    Both engines expose .run(topic), .control and .sid; the graph engine adds .resume(sid).
    """
    budget = budget or Budget()
    holder: dict = {}
    usage_sink = lambda u: store.record_call(  # noqa: E731
        session_id=holder["e"].sid or None, purpose=u.purpose, model=u.model,
        input_tokens=u.input_tokens, output_tokens=u.output_tokens,
        cache_read_tokens=u.cache_read_tokens, cache_write_tokens=u.cache_write_tokens,
        web_searches=u.web_searches, cost_usd=u.cost_usd, stop_reason=u.stop_reason,
        request_id=u.request_id)

    if offline:
        from .fake_llm import FakeLLM
        llm = FakeLLM(usage_sink)
    else:
        llm = (backend or Backend()).make_llm(usage_sink)

    if engine == "graph":
        from .graph import ResearchGraph
        eng = ResearchGraph(llm, store, ui, checkpoint_path=home / "checkpoints.db",
                            budget=budget, reports_dir=home / "reports", use_profile=use_profile,
                            source_policy=source_policy)
    else:
        eng = Orchestrator(llm, store, ui, budget=budget, reports_dir=home / "reports",
                           use_profile=use_profile, source_policy=source_policy)
    holder["e"] = eng
    return eng


def build_engine(args: argparse.Namespace, store: Store, engine: str):
    ui = TerminalUI(auto_approve=getattr(args, "yes", False),
                    verbose=getattr(args, "verbose", False))
    budget = Budget(max_rounds=args.rounds, max_tasks=args.max_tasks,
                    max_parallel=args.parallel, max_searches=args.searches,
                    verify_claims=0 if getattr(args, "no_verify", False)
                    else getattr(args, "verify_claims", 12))
    backend = backend_from_args(args).validate()
    policy = None
    if getattr(args, "block", None) or getattr(args, "only", None):  # per-run extras + saved rules
        policy = SourcePolicy.from_rules(store.source_rules(), block=tuple(args.block or ()),
                                         only=tuple(args.only or ()))
    eng = create_engine(store, ui, engine=engine, offline=args.offline, budget=budget,
                        backend=backend, use_profile=not getattr(args, "no_profile", False),
                        source_policy=policy)
    if not args.offline:
        console.print(f"[dim]Model: {backend.label}[/]")
    install_pause_handler(eng, ui)
    return eng


def install_pause_handler(engine, ui: TerminalUI) -> None:
    """First Ctrl-C requests a pause at the next checkpoint; a second one exits."""
    presses = {"n": 0}

    def on_sigint(signum, frame):
        presses["n"] += 1
        if presses["n"] >= 2:
            console.print("\n[red]Exiting.[/]")
            os._exit(130)
        console.print("\n[yellow]Pause requested — will stop at the next checkpoint "
                      "(Ctrl-C again to quit).[/]")
        engine.control.request_pause()

    signal.signal(signal.SIGINT, on_sigint)
    original_override = ui.override

    def override_and_reset(plan):
        presses["n"] = 0
        return original_override(plan)

    ui.override = override_and_reset  # type: ignore[method-assign]


def print_report(report) -> int:
    if report:
        console.print(Panel(Markdown(report.to_markdown()), title="Report", border_style="green"))
    return 0 if report else 1


def cmd_research(args: argparse.Namespace, store: Store) -> int:
    if args.follow_up and not store.session(args.follow_up):
        console.print(f"[red]Unknown session {args.follow_up}[/] (see rootlogic history)")
        return 1
    engine = build_engine(args, store, args.engine)
    prompt = "What should I dig into next?" if args.follow_up else "What should I research?"
    topic = " ".join(args.topic) or Prompt.ask(f"[bold]{prompt}[/]")
    return print_report(engine.run(topic, parent=args.follow_up))


def cmd_resume(args: argparse.Namespace, store: Store) -> int:
    engine = build_engine(args, store, "graph")
    try:
        return print_report(engine.resume(args.session))
    except ValueError as e:
        console.print(f"[red]{e}[/] (only sessions started with --engine graph can resume)")
        return 1


def cmd_web(args: argparse.Namespace, store: Store) -> int:
    try:
        from .web import serve
    except ImportError:
        console.print("[red]Web UI needs extra packages:[/] pip install -e '.[web]'")
        return 1
    if args.host not in ("127.0.0.1", "localhost"):
        console.print("[yellow]Warning: the web UI has no authentication and spends your API "
                      "credits. Only expose it on a network you trust.[/]")
    console.print(f"rootlogic web UI → http://{args.host}:{args.port}  (Ctrl-C to stop)")
    backend = backend_from_args(args).validate()
    console.print(f"[dim]Model: {backend.label}[/]")
    serve(args.db, HOME, host=args.host, port=args.port, backend=backend)
    return 0


def cmd_graph(args: argparse.Namespace, store: Store) -> int:
    from .graph import ResearchGraph
    from .fake_llm import FakeLLM
    g = ResearchGraph(FakeLLM(), store, TerminalUI(), checkpoint_path=":memory:")
    console.print(g.mermaid(), highlight=False, markup=False)
    return 0


def cmd_history(args, store: Store) -> int:
    table = Table(title="Research sessions")
    for col in ("id", "when", "status", "topic", "follows", "cost"):
        table.add_column(col)
    for s in store.sessions(args.limit):
        u = store.usage(s["id"])
        table.add_row(s["id"], s["created_at"][:16].replace("T", " "), s["status"], s["topic"],
                      s.get("parent_id") or "", f"${u['cost_usd']:.2f}")
    console.print(table)
    if sugg := store.suggestions():
        console.print("[bold]Suggested next topics:[/] " + " · ".join(sugg))
    return 0


def cmd_eval(args, store: Store) -> int:
    """Run the evaluation set and score the results (see rootlogic/evaluate.py)."""
    import tempfile

    from .evaluate import EvalRun, compare, load_cases, run_eval

    cases = load_cases(args.cases, ids=args.case, kinds=args.kind)
    if not cases:
        console.print("[red]No matching cases.[/]")
        return 1
    backend = backend_from_args(args).validate()
    if not args.offline and not args.yes:
        console.print(f"About to run [bold]{len(cases)}[/] live research sessions on "
                      f"{backend.label}. Each costs roughly as much as a normal research run.")
        if Prompt.ask("Continue?", choices=["y", "n"], default="n") != "y":
            return 1
    budget = Budget(max_rounds=args.rounds, max_tasks=args.max_tasks, max_parallel=args.parallel,
                    max_searches=args.searches, verify_claims=args.verify_claims)
    work = Path(tempfile.mkdtemp(prefix="rootlogic-eval-"))

    def make_engine(case_store, ui):
        return create_engine(case_store, ui, engine=args.engine, offline=args.offline,
                             budget=budget, home=work, backend=backend, use_profile=False,
                             source_policy=SourcePolicy())

    def show(r):
        mark = "[green]pass[/]" if r.passed else "[red]fail[/]"
        extra = "" if r.passed else " — " + "; ".join(r.reasons)
        console.print(f"{mark} {r.kind:<10} {r.id}{escape(extra)}", highlight=False)

    run = run_eval(cases, make_engine, engine_name=args.engine,
                   model="offline-fake" if args.offline else backend.label, on_result=show)
    out = Path(args.out or f"evals/results/{run.started_at.replace(':', '')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(run.model_dump_json(indent=2))

    table = Table(title="Evaluation summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    diff = compare(run, EvalRun.model_validate_json(Path(args.baseline).read_text())) \
        if args.baseline else None
    for k, v in run.summary.items():
        delta = diff["deltas"].get(k) if diff else None
        table.add_row(k, f"{v}" + (f"  ({delta:+})" if delta else ""))
    console.print(table)
    if diff and diff["changed"]:
        console.print("[bold]Changed vs baseline:[/] " + ", ".join(
            f"{k}: {v}" for k, v in diff["changed"].items()))
    console.print(f"Results saved to {out}")
    return 0


def cmd_sources(args, store: Store) -> int:
    """Your rules for which sites research may use and how much to trust them."""
    if args.action in RULES:
        if not args.domains:
            console.print(f"[red]Usage:[/] rootlogic sources {args.action} <domain> [...]")
            return 1
        for d in args.domains:
            store.set_source_rule(clean_domain(d), args.action)
    elif args.action == "rm":
        for d in args.domains:
            gone = store.remove_source_rule(clean_domain(d))
            console.print(f"{clean_domain(d)}: {'removed' if gone else 'no rule'}")
    rules = store.source_rules()
    if not rules:
        console.print("[dim]No source rules. Examples: rootlogic sources block example.com · "
                      "sources trust who.int · sources allow nature.com science.org "
                      "(allow = only these sites)[/]")
        return 0
    table = Table(title="Source rules")
    table.add_column("domain")
    table.add_column("rule")
    meaning = {"block": "never used", "allow": "allowlist: only allowed sites are used",
               "trust": "credibility forced high", "distrust": "credibility forced low; "
               "can't be a claim's only support"}
    for r in rules:
        table.add_row(r["domain"], f"{r['rule']} — {meaning[r['rule']]}")
    console.print(table)
    return 0


def cmd_profile(args, store: Store) -> int:
    """View and edit the standing preferences the assistant has learned."""
    if args.action == "add":
        text = " ".join(args.args).strip()
        if not text:
            console.print("[red]Usage:[/] rootlogic profile add <preference text>")
            return 1
        added = store.add_preference(args.category, text)
        console.print("Added." if added else "Already in your profile.")
    elif args.action == "rm":
        ids = [a for a in args.args if a.isdigit()]
        if not ids:
            console.print("[red]Usage:[/] rootlogic profile rm <id> [<id> ...]")
            return 1
        for i in ids:
            console.print(f"{i}: " + ("removed" if store.remove_preference(int(i)) else "not found"))
    elif args.action == "clear":
        console.print(f"Removed {store.clear_preferences()} preference(s).")
    prefs = store.preferences()
    if not prefs:
        console.print("[dim]Profile is empty. It fills in as you answer questions and leave "
                      "notes, or add entries with: rootlogic profile add <text>[/]")
        return 0
    table = Table(title="Your research profile (standing preferences)")
    for col in ("id", "category", "preference", "learned from"):
        table.add_column(col)
    for p in prefs:
        table.add_row(str(p["id"]), p["category"], p["text"], p["session_id"] or "added by you")
    console.print(table)
    return 0


def cmd_show(args, store: Store) -> int:
    s = store.session(args.session)
    if not s or not s["report_path"] or not Path(s["report_path"]).exists():
        console.print("[red]No report for that session.[/]")
        return 1
    console.print(Markdown(Path(s["report_path"]).read_text()))
    return 0


def cmd_log(args, store: Store) -> int:
    for e in store.events(args.session):
        console.print(f"[dim]{e['ts'][11:19]}[/] [cyan]{e['type']:<18}[/] {escape(e['message'])}",
                      highlight=False)
    console.print()
    for m in store.messages(args.session):
        console.print(f"[bold]{m['role']}[/] ({m['kind']}): {escape(m['content'])}", highlight=False)
    return 0


def cmd_usage(args, store: Store) -> int:
    if args.session:
        table = Table(title=f"Token usage · session {args.session}")
        for col in ("purpose", "calls", "input", "output", "searches", "cost"):
            table.add_column(col, justify="right" if col != "purpose" else "left")
        for r in store.usage_by_purpose(args.session):
            table.add_row(r["purpose"], str(r["calls"]), f"{r['input_tokens']:,}",
                          f"{r['output_tokens']:,}", str(r["web_searches"]), f"${r['cost_usd']:.3f}")
        console.print(table)
    u = store.usage(args.session)
    console.print(f"Total: {u['calls']} calls · in {u['input_tokens']:,} · out "
                  f"{u['output_tokens']:,} · cache read {u['cache_read_tokens']:,} · "
                  f"searches {u['web_searches']} · [bold]${u['cost_usd']:.2f}[/]")
    return 0


def cmd_forget(args, store: Store) -> int:
    store.delete_session(args.session)
    console.print(f"Deleted session {args.session} and its memory.")
    return 0


def add_backend_args(p: argparse.ArgumentParser) -> None:
    """Model + search flags, shared by research, resume and web."""
    g = p.add_argument_group("model and search")
    g.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                   help="anthropic (Claude) or openai: any OpenAI-compatible server "
                        "(OpenAI, Ollama, LM Studio, vLLM, OpenRouter)")
    g.add_argument("--model", help="model id; default claude-opus-5, required for openai")
    g.add_argument("--base-url", help="OpenAI-compatible server, e.g. http://localhost:11434/v1")
    g.add_argument("--no-strict", action="store_true",
                   help="openai provider: for servers without strict JSON-schema support")
    g.add_argument("--prices", metavar="IN,OUT",
                   help="openai provider: USD per million input,output tokens for cost tracking")
    g.add_argument("--search", choices=["anthropic", "tavily"], default="anthropic",
                   help="web search for sub-agents: Claude's built-in tools, or Tavily via our "
                        "own SearchProvider tools (needs TAVILY_API_KEY)")
    g.add_argument("--zdr", action="store_true",
                   help="Zero Data Retention: Claude web tools without dynamic filtering")


def add_quality_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("quality and sources")
    g.add_argument("--verify-claims", type=int, default=12, metavar="N",
                   help="claims to check against their cited pages (default 12)")
    g.add_argument("--no-verify", action="store_true", help="skip claim verification")
    g.add_argument("--block", action="append", metavar="DOMAIN",
                   help="never use this site in this run (repeatable)")
    g.add_argument("--only", action="append", metavar="DOMAIN",
                   help="use only these sites in this run (repeatable)")


def backend_from_args(args: argparse.Namespace) -> Backend:
    return Backend(provider=getattr(args, "provider", "anthropic"),
                   model=getattr(args, "model", None), base_url=getattr(args, "base_url", None),
                   search=getattr(args, "search", "anthropic"), zdr=getattr(args, "zdr", False),
                   strict=not getattr(args, "no_strict", False),
                   prices=parse_prices(getattr(args, "prices", None)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rootlogic", description="Agentic personal research assistant")
    p.add_argument("--db", default=str(HOME / "rootlogic.db"))
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("research", help="research a topic")
    r.add_argument("topic", nargs="*")
    r.add_argument("-y", "--yes", action="store_true", help="auto-approve the plan")
    r.add_argument("--follow-up", metavar="SESSION",
                   help="continue an earlier session: reuse its findings, research only what's new")
    r.add_argument("--no-profile", action="store_true",
                   help="don't use or update your standing preferences for this run")
    add_quality_args(r)
    r.add_argument("-v", "--verbose", action="store_true", help="show dropped sources")
    r.add_argument("--offline", action="store_true", help="use the fake LLM (no API calls)")
    r.add_argument("--rounds", type=int, default=2, help="max reflection rounds")
    r.add_argument("--max-tasks", type=int, default=10)
    r.add_argument("--parallel", type=int, default=4)
    r.add_argument("--searches", type=int, default=5, help="web searches per sub-agent")
    add_backend_args(r)
    r.add_argument("--engine", choices=["loop", "graph"], default="loop",
                   help="loop: hand-rolled orchestrator · graph: LangGraph (resumable)")
    r.set_defaults(fn=cmd_research)

    rs = sub.add_parser("resume", help="continue a --engine graph session from its checkpoint")
    rs.add_argument("session")
    rs.add_argument("--offline", action="store_true")
    add_backend_args(rs)
    add_quality_args(rs)
    rs.set_defaults(fn=cmd_resume, rounds=2, max_tasks=10, parallel=4, searches=5)

    w = sub.add_parser("web", help="start the web UI (FastAPI + SSE)")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    add_backend_args(w)
    w.set_defaults(fn=cmd_web)

    gr = sub.add_parser("graph", help="print the LangGraph engine as a Mermaid diagram")
    gr.set_defaults(fn=cmd_graph)

    ev = sub.add_parser("eval", help="run the evaluation set and score quality and safety")
    ev.add_argument("--cases", default="evals/cases.json")
    ev.add_argument("--case", action="append", help="run only this case id (repeatable)")
    ev.add_argument("--kind", action="append",
                    choices=["fact", "hoax", "contested", "recency", "harmful"])
    ev.add_argument("--offline", action="store_true", help="fake LLM: tests the harness, free")
    ev.add_argument("--engine", choices=["loop", "graph"], default="loop")
    ev.add_argument("--out", help="where to save results JSON")
    ev.add_argument("--baseline", help="earlier results JSON to compare against")
    ev.add_argument("-y", "--yes", action="store_true", help="skip the cost confirmation")
    ev.add_argument("--rounds", type=int, default=1)
    ev.add_argument("--max-tasks", type=int, default=6)
    ev.add_argument("--parallel", type=int, default=4)
    ev.add_argument("--searches", type=int, default=4)
    ev.add_argument("--verify-claims", type=int, default=12)
    add_backend_args(ev)
    ev.set_defaults(fn=cmd_eval)

    so = sub.add_parser("sources", help="block, allow, trust or distrust websites")
    so.add_argument("action", nargs="?", choices=["list", *RULES, "rm"], default="list")
    so.add_argument("domains", nargs="*")
    so.set_defaults(fn=cmd_sources)

    pr = sub.add_parser("profile", help="view or edit your standing preferences")
    pr.add_argument("action", nargs="?", choices=["list", "add", "rm", "clear"], default="list")
    pr.add_argument("args", nargs="*")
    pr.add_argument("--category", default="other",
                    choices=["audience", "region", "time_window", "sources_prefer",
                             "sources_avoid", "format", "expertise", "other"])
    pr.set_defaults(fn=cmd_profile)

    h = sub.add_parser("history", help="list past sessions and suggested topics")
    h.add_argument("--limit", type=int, default=20)
    h.set_defaults(fn=cmd_history)

    for name, fn, helptext in (("show", cmd_show, "print a session's report"),
                               ("log", cmd_log, "print a session's action log and conversation"),
                               ("forget", cmd_forget, "delete a session and its memory")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("session")
        sp.set_defaults(fn=fn)

    u = sub.add_parser("usage", help="token usage and cost")
    u.add_argument("session", nargs="?")
    u.set_defaults(fn=cmd_usage)

    args = p.parse_args(argv)
    from .search import SearchError
    try:
        return args.fn(args, Store(args.db))
    except SearchError as e:
        console.print(f"[red]Search provider error:[/] {e}")
        return 2
    except BackendError as e:
        console.print(f"[red]Invalid model settings:[/] {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
