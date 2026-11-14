"""
Four code extractors for the FORGE Politeness-Tax LiveCodeBench study, as one runnable
module. Each takes (completion: str, entry_point: str) and returns the code string it would
hand to the scorer (or None / "" when it recovers nothing).

  extract_naive    - pre-fix extractor: first fenced block naming the target, else the whole
                     fence-stripped reply (trailing prose and all). Fails on unfenced
                     code+prose replies (SyntaxError). This is the artifact source.
  extract_solution - "ours" / Sieve: trim-to-compilable, keep helpers, shape-agnostic. VERBATIM
                     copy of score_lcb.py::extract_solution (+ _trim_to_compilable); the scoring
                     harness imports its own copy, so the two are identical.
  extract_sanitize - EvalPlus sanitize (vendored byte-identical from evalplus/sanitize.py:
                     code_extract + tree-sitter reachability). Needs tree_sitter + tree_sitter_python.
  extract_lcb      - LiveCodeBench official extract_code (lcb_runner/utils/extraction_utils.py,
                     generic/API branch, VERBATIM): text between the last two triple-backtick fence lines; ""
                     if fewer than 2 fence lines (unfenced replies -> empty -> no-code FAIL).

Binding extracted code to the entry point for execution is done by
score_lcb.py::resolve_callable (shape-agnostic) in the scoring harness, not here.

Self-test:  python extractors.py       (runs all four on one stored reply)
"""
from __future__ import annotations  # sanitize's type hints (Node, Generator) stay lazy,
                                     # so naive/ours/lcb import fine without tree_sitter.
import ast
import re  # noqa: F401  (used by the extractors below)

# ===================== ours / Sieve (verbatim from score_lcb.py) =====================
def _trim_to_compilable(code):
    """Drop trailing lines until the source compiles. Conversational replies put a
    prose paragraph AFTER the code ('This checks every adjacent pair...'); with no
    fences to delimit it, that prose used to be exec'd and threw SyntaxError,
    failing correct code. The polite/detailed framing elicits more explain-after-
    code, so this bug penalized it hardest — a scoring artifact, not correctness.
    We keep the largest leading prefix that is valid Python and still has the
    class/method.

    We check with compile(...,'exec'), not ast.parse: the scorer runs exec(), and a
    few errors (a stray top-level return/break/continue/yield — e.g. a model that
    appends `return Solution().f` as a chat-style 'here is the answer') pass
    ast.parse but fail at compile time, so an ast.parse predicate would keep code the
    scorer then rejects. compile() compiles without executing, so it has no side
    effects while matching exec()'s strictness."""
    lines = code.split("\n")
    while lines:
        src = "\n".join(lines)
        try:
            compile(src, "<lcb>", "exec")
            return src
        except SyntaxError:
            lines.pop()
    return None


def extract_solution(completion, entry):
    """Pull runnable code that defines `class Solution` (or the method), robust to
    trailing prose, leading chatter, AND helper definitions (Fenwick/DSU/Segments…)
    the model defines BEFORE `class Solution`. Returns the first candidate that
    compiles and still defines the target. Candidates that KEEP the helpers are
    tried first, so we don't hand exec() a Solution that references an undefined
    helper we chopped off."""
    if not completion:
        return None
    text = "\n".join(l for l in completion.split("\n") if not l.strip().startswith("```"))
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", completion, re.DOTALL)
    code_blocks = [b for b in blocks if re.search(r"(?m)^\s*(?:class|def|import|from|@)\s", b)]

    cands = []
    # (B) from the FIRST top-level code construct to the end — keeps helper
    #     classes/functions defined above `class Solution` (fixes the drop bug).
    m = re.search(r"(?m)^(?:class|def|import|from|@)\s", text)
    if m:
        cands.append(text[m.start():])
    # (A) all fenced code blocks concatenated — helper and Solution in separate blocks.
    if code_blocks:
        cands.append("\n\n".join(code_blocks))
    # (C) a single block that names the target (old high-confidence path).
    for b in blocks:
        if "class Solution" in b or f"def {entry}" in b:
            cands.append(b)
    # (D) old anchor slice, last resort.
    for anchor in ("class Solution", f"def {entry}"):
        i = text.find(anchor)
        if i != -1:
            cands.append(text[i:])

    for c in cands:
        t = _trim_to_compilable(c)
        if t and ("class Solution" in t or f"def {entry}" in t):
            return t
    return cands[0] if cands else None


# ===================== naive (pre-fix) =====================
def extract_naive(completion, entry):
    """Pre-fix extractor: first fenced block naming the target, else the whole
    fence-stripped completion -- trailing prose and all (-> SyntaxError on correct code)."""
    if not completion:
        return None
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", completion, re.DOTALL)
    for b in blocks:
        if "class Solution" in b or f"def {entry}" in b:
            return b
    if "class Solution" in completion or f"def {entry}" in completion:
        return "\n".join(l for l in completion.split("\n") if not l.strip().startswith("```"))
    return None


# ===================== EvalPlus sanitize (vendored) =====================
# Vendored EvalPlus sanitize() -- the byte-identical functions from evalplus/sanitize.py,
# stripped of the evalplus.data/tqdm/fire CLI deps so it needs ONLY tree_sitter + the
# Python grammar. This IS EvalPlus's sanitize algorithm (code_extract + reachability).
import re, ast
from typing import Dict, Generator, List, Optional, Set, Tuple
try:
    import tree_sitter_python
    from tree_sitter import Language, Node, Parser
    _TREE_SITTER_OK = True
except ImportError:  # extract_sanitize will raise; the other three extractors still work
    _TREE_SITTER_OK = False

def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        return False

CLASS_TYPE = "class_definition"
FUNCTION_TYPE = "function_definition"
IMPORT_TYPE = ["import_statement", "import_from_statement"]
IDENTIFIER_TYPE = "identifier"
ATTRIBUTE_TYPE = "attribute"
RETURN_TYPE = "return_statement"
EXPRESSION_TYPE = "expression_statement"
ASSIGNMENT_TYPE = "assignment"


def code_extract(text: str) -> str:
    # Remove ANSI escape sequences, which was caused by ollama (small qwen3 models, but may be others as well)
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    text = ansi_escape.sub("", text)

    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])


def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:
    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
        for child in node.children:
            if child.type == IDENTIFIER_TYPE:
                deps.add(child.text.decode("utf8"))
            else:
                dfs_get_deps(child, deps)

    name2deps = {}
    for name, node in nodes:
        deps = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str, call_graph: Dict[str, str]) -> Set[str]:
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if not (neighbour in visited):
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def get_definition_name(node: Node) -> str:
    for child in node.children:
        if child.type == IDENTIFIER_TYPE:
            return child.text.decode("utf8")


def traverse_tree(node: Node) -> Generator[Node, None, None]:
    cursor = node.walk()
    depth = 0

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def has_return_statement(node: Node) -> bool:
    traverse_nodes = traverse_tree(node)
    for node in traverse_nodes:
        if node.type == RETURN_TYPE:
            return True
    return False


def extract_target_code_or_empty(code: str, entrypoint: Optional[str] = None) -> str:
    code = code_extract(code)
    code_bytes = bytes(code, "utf8")
    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(code_bytes)
    class_names = set()
    function_names = set()
    variable_names = set()

    root_node = tree.root_node
    import_nodes = []
    definition_nodes = []

    for child in root_node.children:
        if child.type in IMPORT_TYPE:
            import_nodes.append(child)
        elif child.type == CLASS_TYPE:
            name = get_definition_name(child)
            if not (
                name in class_names or name in variable_names or name in function_names
            ):
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == FUNCTION_TYPE:
            name = get_definition_name(child)
            if not (
                name in function_names or name in variable_names or name in class_names
            ) and has_return_statement(child):
                definition_nodes.append((name, child))
                function_names.add(get_definition_name(child))
        elif (
            child.type == EXPRESSION_TYPE and child.children[0].type == ASSIGNMENT_TYPE
        ):
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if not (
                name in variable_names or name in function_names or name in class_names
            ):
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    if entrypoint:
        name2deps = get_deps(definition_nodes)
        reacheable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = b""

    for node in import_nodes:
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"

    for pair in definition_nodes:
        name, node = pair
        if entrypoint and not (name in reacheable):
            continue
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"
    return sanitized_output[:-1].decode("utf8")


def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    sanitized_code = extract_target_code_or_empty(code, entrypoint).strip()
    if not sanitized_code:
        return code_extract(code)
    return sanitized_code


def extract_sanitize(completion, entry):
    if not _TREE_SITTER_OK:
        raise RuntimeError('extract_sanitize needs tree_sitter + tree_sitter_python: pip install tree_sitter tree_sitter_python')
    try:
        return sanitize(completion or '', entrypoint=entry)
    except Exception as e:
        return None


# ===================== LiveCodeBench official extract_code =====================
def extract_lcb(completion, entry):
    """LiveCodeBench official extract_code (lcb_runner/utils/extraction_utils.py), generic/API
    branch, VERBATIM: text between the LAST TWO ``` fence lines; "" if < 2 fence lines. Fence-only,
    so 'no fences' replies yield nothing. `entry` is unused (the official fn does not anchor on the
    target); kept for a uniform signature."""
    if completion is None:
        return ""
    outputlines = completion.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])

# ===================== registry + self-test =====================
extract_ours = extract_solution  # alias used throughout the paper ("ours" / Sieve)

EXTRACTORS = {
    "naive": extract_naive,
    "ours": extract_ours,
    "sanitize": extract_sanitize,
    "lcb": extract_lcb,
}

if __name__ == "__main__":
    import json, os
    here = os.path.dirname(os.path.abspath(__file__))
    resp = json.load(open(os.path.join(here, "data", "lcb_v3_responses.json")))
    # lcb/3747, detailed, ChatGPT: correct code + trailing prose, unfenced (paper's figure example)
    r = next(x for x in resp if x["task_id"] == "lcb/3747"
             and x["model"] == "chatgpt" and x["level"] == "webchat_detailed")
    c, entry = r["completion"], r["entry_point"]
    print("reply:", r["task_id"], r["model"], r["level"], "(fences=%d)\n" % c.count("`" * 3))
    for name, fn in EXTRACTORS.items():
        code = fn(c, entry)
        if not code:
            verdict = "no code recovered -> FAIL"
        else:
            try:
                ast.parse(code)
                verdict = "parses OK"
            except SyntaxError as e:
                verdict = "SyntaxError line %s: %s  -> FAIL" % (e.lineno, e.msg)
        print("  %-9s -> %s" % (name, verdict))

