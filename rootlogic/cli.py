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
                  budget: Budget | None = None, home: Path = HOME, zdr: bool = False):
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
        from .llm import AnthropicLLM
        llm = AnthropicLLM(usage_sink, zdr=zdr)

    if engine == "graph":
        from .graph import ResearchGraph
        eng = ResearchGraph(llm, store, ui, checkpoint_path=home / "checkpoints.db",
                            budget=budget, reports_dir=home / "reports")
    else:
        eng = Orchestrator(llm, store, ui, budget=budget, reports_dir=home / "reports")
    holder["e"] = eng
    return eng


def build_engine(args: argparse.Namespace, store: Store, engine: str):
    ui = TerminalUI(auto_approve=getattr(args, "yes", False),
                    verbose=getattr(args, "verbose", False))
    budget = Budget(max_rounds=args.rounds, max_tasks=args.max_tasks,
                    max_parallel=args.parallel, max_searches=args.searches)
    eng = create_engine(store, ui, engine=engine, offline=args.offline, budget=budget,
                        zdr=getattr(args, "zdr", False))
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
    engine = build_engine(args, store, args.engine)
    topic = " ".join(args.topic) or Prompt.ask("[bold]What should I research?[/]")
    return print_report(engine.run(topic))


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
    serve(args.db, HOME, host=args.host, port=args.port, zdr=args.zdr)
    return 0


def cmd_graph(args: argparse.Namespace, store: Store) -> int:
    from .graph import ResearchGraph
    from .fake_llm import FakeLLM
    g = ResearchGraph(FakeLLM(), store, TerminalUI(), checkpoint_path=":memory:")
    console.print(g.mermaid(), highlight=False, markup=False)
    return 0


def cmd_history(args, store: Store) -> int:
    table = Table(title="Research sessions")
    for col in ("id", "when", "status", "topic", "cost"):
        table.add_column(col)
    for s in store.sessions(args.limit):
        u = store.usage(s["id"])
        table.add_row(s["id"], s["created_at"][:16].replace("T", " "), s["status"], s["topic"],
                      f"${u['cost_usd']:.2f}")
    console.print(table)
    if sugg := store.suggestions():
        console.print("[bold]Suggested next topics:[/] " + " · ".join(sugg))
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rootlogic", description="Agentic personal research assistant")
    p.add_argument("--db", default=str(HOME / "rootlogic.db"))
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("research", help="research a topic")
    r.add_argument("topic", nargs="*")
    r.add_argument("-y", "--yes", action="store_true", help="auto-approve the plan")
    r.add_argument("-v", "--verbose", action="store_true", help="show dropped sources")
    r.add_argument("--offline", action="store_true", help="use the fake LLM (no API calls)")
    r.add_argument("--rounds", type=int, default=2, help="max reflection rounds")
    r.add_argument("--max-tasks", type=int, default=10)
    r.add_argument("--parallel", type=int, default=4)
    r.add_argument("--searches", type=int, default=5, help="web searches per sub-agent")
    r.add_argument("--zdr", action="store_true",
                   help="Zero Data Retention: web tools without dynamic filtering")
    r.add_argument("--engine", choices=["loop", "graph"], default="loop",
                   help="loop: hand-rolled orchestrator · graph: LangGraph (resumable)")
    r.set_defaults(fn=cmd_research)

    rs = sub.add_parser("resume", help="continue a --engine graph session from its checkpoint")
    rs.add_argument("session")
    rs.add_argument("--offline", action="store_true")
    rs.add_argument("--zdr", action="store_true",
                    help="Zero Data Retention: web tools without dynamic filtering")
    rs.set_defaults(fn=cmd_resume, rounds=2, max_tasks=10, parallel=4, searches=5)

    w = sub.add_parser("web", help="start the web UI (FastAPI + SSE)")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--zdr", action="store_true",
                   help="Zero Data Retention: web tools without dynamic filtering")
    w.set_defaults(fn=cmd_web)

    gr = sub.add_parser("graph", help="print the LangGraph engine as a Mermaid diagram")
    gr.set_defaults(fn=cmd_graph)

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
    return args.fn(args, Store(args.db))


if __name__ == "__main__":
    sys.exit(main())
