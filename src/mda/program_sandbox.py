"""Sandbox for LLM-written experiment programs (ported from ``expprog.llm.sandbox``).

A program is the source of one class with a single method::

    class Name(Program):
        def step(self, history, belief):
            return next_input

Two layers keep it contained:

1. **AST whitelist (in this process, before anything runs).** Literals, names,
   arithmetic, comparisons, boolean and conditional expressions, tuples, lists,
   subscripts, ``if``, ``for`` over ``range(...)``, assignments and ``return``.
   Calls only to ``min, max, abs, len, range, float, int, round, any, all, sum``,
   public ``math`` functions and ``history.obs_at``/``history.last_obs``.
   Attributes only ``history.inputs/obs/t/horizon/obs_at/last_obs`` and public
   ``math`` names. Rejected: imports, ``while``, other ``def``s, lambdas and
   comprehensions, ``try``/``with``/``raise``, dict and set literals, f-strings,
   names starting with ``_``, and every other attribute or call. With no imports,
   no underscore names and no method calls on values, the source has no handle to
   the filesystem, the network or interpreter internals.
2. **A separate worker process.** Unlike ``expprog``, vetted source is never
   executed in this process. A worker (``python -I -S``) with CPU and memory
   rlimits, ``os``/``subprocess``/``socket``/... poisoned in ``sys.modules`` and
   ``__builtins__`` replaced by the whitelist executes it. A ``ProgramSession``
   sends the worker a *batch* of histories per step and receives one input per
   history, so closed-loop scoring costs one round trip per episode step, not
   one per imagined history, and still calls ``step`` on each history separately
   (programs cannot couple histories).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import select
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from .programs import History, Program, ProgramError


SAFE_CALLS = frozenset({"min", "max", "abs", "len", "range", "float", "int", "round", "any", "all", "sum"})
HISTORY_ATTRIBUTES = frozenset({"inputs", "obs", "t", "horizon", "obs_at", "last_obs"})
HISTORY_METHODS = frozenset({"obs_at", "last_obs"})
_FORBIDDEN_NAMES = frozenset({
    "exec", "eval", "compile", "open", "globals", "locals", "getattr", "setattr", "delattr",
    "vars", "dir", "input", "breakpoint", "type", "object", "super", "help", "memoryview",
})
_ALLOWED = (
    ast.Module, ast.ClassDef, ast.FunctionDef, ast.Return, ast.If, ast.For, ast.Assign,
    ast.AugAssign, ast.Pass, ast.Expr, ast.Break, ast.Continue,
    ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Tuple, ast.List, ast.Subscript, ast.Slice, ast.Call, ast.Attribute,
    ast.Load, ast.Store, ast.arguments, ast.arg,
    ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
)


class SandboxError(ValueError):
    """Source rejected by the whitelist or by the worker."""


def _check(node: ast.AST) -> None:
    if not isinstance(node, _ALLOWED):
        raise SandboxError(f"{type(node).__name__} is not allowed")
    if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, str, bool, type(None))):
        raise SandboxError("only numeric, string, boolean and None constants are allowed")
    if isinstance(node, ast.Name):
        if node.id.startswith("_") or node.id in _FORBIDDEN_NAMES:
            raise SandboxError(f"name {node.id!r} is not allowed")
        if node.id == "self":
            raise SandboxError("programs are stateless: self may not be used")
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("_") or not isinstance(node.value, ast.Name):
            raise SandboxError(f"attribute {node.attr!r} is not allowed")
        if node.value.id == "math":
            if not hasattr(math, node.attr):
                raise SandboxError(f"math.{node.attr} does not exist")
        elif node.value.id != "history" or node.attr not in HISTORY_ATTRIBUTES:
            raise SandboxError(f"attribute {node.value.id}.{node.attr} is not allowed")
        if not isinstance(node.ctx, ast.Load):
            raise SandboxError("attributes may not be assigned")
    if isinstance(node, ast.Call):
        if node.keywords:
            raise SandboxError("keyword arguments are not allowed")
        function = node.func
        allowed = (
            (isinstance(function, ast.Name) and function.id in SAFE_CALLS)
            or (isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name)
                and (function.value.id == "math"
                     or (function.value.id == "history" and function.attr in HISTORY_METHODS)))
        )
        if not allowed:
            raise SandboxError("only whitelisted functions, math.* and history.obs_at/last_obs may be called")
    if isinstance(node, ast.For):
        iterator = node.iter
        if not (isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Name)
                and iterator.func.id == "range") or node.orelse:
            raise SandboxError("for loops may only iterate over range(...), without else")
    if isinstance(node, ast.ClassDef):
        if node.decorator_list or node.keywords or [
            base for base in node.bases if not (isinstance(base, ast.Name) and base.id == "Program")
        ]:
            raise SandboxError("the class may only subclass Program, without decorators")
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.Pass)) and not (
                isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant)
            ):
                raise SandboxError("the class body may contain only a docstring and step()")
    if isinstance(node, ast.FunctionDef):
        arguments = node.args
        if (node.name != "step" or node.decorator_list or node.returns is not None
                or arguments.vararg or arguments.kwarg or arguments.posonlyargs
                or arguments.kwonlyargs or arguments.defaults
                or any(a.annotation is not None for a in arguments.args)
                or [a.arg for a in arguments.args] != ["self", "history", "belief"]):
            raise SandboxError("define only step(self, history, belief), with no defaults or annotations")
    for child in ast.iter_child_nodes(node):
        # ``self`` is the method's own first argument; it is declared, not used.
        if isinstance(node, ast.arguments) and isinstance(child, ast.arg):
            continue
        _check(child)


def canonical_source(source: str) -> str:
    """Parse, whitelist, and return the canonical (``ast.unparse``) source."""
    if not isinstance(source, str) or not source.strip():
        raise SandboxError("empty source")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise SandboxError(f"syntax error: {error}") from error
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1 or len(tree.body) != 1:
        raise SandboxError("the source must be exactly one class definition")
    if not any(isinstance(item, ast.FunctionDef) for item in classes[0].body):
        raise SandboxError("the class must define step(self, history, belief)")
    _check(tree)
    return ast.unparse(tree)


_WORKER = r'''
import json, math, resource, sys
cpu, memory = int(sys.argv[1]), int(sys.argv[2])
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
try:
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
except (ValueError, OSError):
    pass
for name in ("os", "subprocess", "socket", "shutil", "ctypes", "importlib", "pickle",
             "runpy", "signal", "threading", "multiprocessing", "io", "pathlib"):
    sys.modules[name] = None
_builtins = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
_range = _builtins["range"]
def capped_range(*arguments):
    value = _range(*arguments)
    if len(value) > 100000:
        raise ValueError("range longer than 100000")
    return value
safe = {name: _builtins[name] for name in
        ("min", "max", "abs", "len", "float", "int", "round", "any", "all", "sum")}
safe.update(range=capped_range, __build_class__=_builtins["__build_class__"],
            __name__="program")
class Program(object):
    pass
class History(object):
    def __init__(self, inputs, obs, horizon):
        self.inputs, self.obs, self.horizon = inputs, obs, horizon
    @property
    def t(self):
        return len(self.inputs)
    def obs_at(self, index):
        return self.obs[index] if 0 <= index < len(self.obs) else None
    def last_obs(self):
        return self.obs[-1] if self.obs else None
def reply(value):
    sys.stdout.write(json.dumps(value) + "\n")
    sys.stdout.flush()
stdin = sys.stdin
namespace = {"__builtins__": safe, "math": math, "Program": Program}
try:
    exec(compile(json.loads(stdin.readline())["source"], "<program>", "exec"), namespace)
    classes = [v for v in namespace.values()
               if isinstance(v, type) and issubclass(v, Program) and v is not Program]
    program = classes[0]()
except Exception as error:
    reply({"ok": False, "error": "definition: %s: %s" % (type(error).__name__, error)})
    sys.exit(0)
reply({"ok": True})
for line in stdin:
    request = json.loads(line)
    outputs = []
    try:
        for inputs, obs in request["histories"]:
            value = program.step(History(inputs, obs, request["horizon"]), request["belief"])
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("step returned %r, not a number" % (value,))
            outputs.append(float(value))
    except Exception as error:
        reply({"ok": False, "error": "step: %s: %s" % (type(error).__name__, error)})
        continue
    reply({"ok": True, "inputs": outputs})
'''


class _WorkerSession:
    def __init__(self, source: str, *, cpu_seconds: int, memory_mb: int, timeout: float) -> None:
        self.timeout = timeout
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", _WORKER, str(cpu_seconds), str(memory_mb * 2**20)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, cwd="/",
        )
        self._request({"source": source})

    def _request(self, message: Mapping[str, Any]) -> dict:
        try:
            self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise ProgramError(f"program worker exited: {error}") from error
        ready, _, _ = select.select([self.process.stdout], [], [], self.timeout)
        line = self.process.stdout.readline() if ready else ""
        if not line:
            self.close()
            if not ready:
                raise ProgramError(f"program step exceeded {self.timeout} s")
            raise ProgramError(f"program worker died (exit {self.process.returncode}; CPU or memory limit?)")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise ProgramError(str(reply.get("error"))[:400])
        return reply

    def __call__(self, histories: Sequence[History], belief: Mapping[str, Any]) -> list[float]:
        reply = self._request({
            "histories": [[list(h.inputs), list(h.obs)] for h in histories],
            "horizon": histories[0].horizon if histories else 0,
            "belief": belief,
        })
        return reply["inputs"]

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            try:
                stream.close()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SandboxedProgram(Program):
    """A vetted program whose ``step`` runs only inside a worker process."""

    def __init__(self, source: str, name: str = "program", *, cpu_seconds: int = 10,
                 memory_mb: int = 512, timeout: float = 30.0) -> None:
        self.source = canonical_source(source)
        self.name = name
        self.cpu_seconds, self.memory_mb, self.timeout = cpu_seconds, memory_mb, timeout
        self.digest = hashlib.sha256(self.source.encode()).hexdigest()[:16]

    @property
    def key(self) -> str:
        return f"program:{self.digest}"

    @property
    def label(self) -> str:
        return f"{self.name}[{self.digest}]"

    def session(self) -> _WorkerSession:
        return _WorkerSession(self.source, cpu_seconds=self.cpu_seconds,
                              memory_mb=self.memory_mb, timeout=self.timeout)


def compile_program(source: str, name: str = "program", **limits) -> SandboxedProgram:
    """Whitelist the source and define it once in a worker; raise ``SandboxError`` if either fails."""
    program = SandboxedProgram(source, name, **limits)
    try:
        with program.session():
            pass
    except ProgramError as error:
        raise SandboxError(str(error)) from error
    return program
