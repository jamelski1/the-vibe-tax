"""Build The_Vibe_Tax_Problem_Breakdown.xlsx — per-problem/per-model success/fail
across HumanEval, HumanEval+, and LiveCodeBench, with the model's actual code."""
import ast, json, os, re
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

import os as _os; ROOT = _os.path.dirname(_os.path.abspath(__file__))
OUT = os.path.join(ROOT, "The_Vibe_Tax_Problem_Breakdown.xlsx")
REP = "agentic_terse"            # representative condition (framing is null)
CAP = 3000                       # max chars for long text cells

# ---------- topic classifier (same as analyze_lcb_capability.py) ----------
TOPICS = [
    ("dynamic programming", r"\b(dynamic programming|dp\b|subsequence|number of ways|minimum cost|maximum (sum|score|value)|partitions?)\b"),
    ("graph", r"\b(graph|node|edge|adjacen|connected component|shortest path|cities|roads?|network)\b"),
    ("tree", r"\b(binary tree|tree|root|leaf|subtree|ancestor|parent node)\b"),
    ("string", r"\b(string|substring|palindrome|character|prefix|suffix|anagram|lexicograph)\b"),
    ("intervals/sorting", r"\b(interval|sort|sorted|merge|schedule|meeting|overlap)\b"),
    ("greedy/array", r"\b(array|subarray|greedy|adjacent|window|two pointers?)\b"),
    ("math/number", r"\b(prime|gcd|lcm|modulo|divisor|digits?|binary representation|factorial|arithmetic)\b"),
    ("bit manipulation", r"\b(bit|xor|bitwise|set bits)\b"),
    ("simulation/geometry", r"\b(simulat|grid|matrix|coordinate|move|direction|snake|robot|game)\b"),
]
def classify(t):
    t = t.lower()
    for name, pat in TOPICS:
        if re.search(pat, t): return name
    return "other"

def lcb_code(completion, entry):
    if not completion: return ""
    cands = []
    for b in re.findall(r"```(?:python|py)?\s*\n(.*?)```", completion, re.DOTALL):
        if "class Solution" in b or f"def {entry}" in b: cands.append(b)
    text = "\n".join(l for l in completion.split("\n") if not l.strip().startswith("```"))
    for anchor in ("class Solution", f"def {entry}"):
        i = text.find(anchor)
        if i != -1: cands.append(text[i:])
    for c in cands:
        lines = c.split("\n")
        while lines:
            src = "\n".join(lines)
            try:
                ast.parse(src)
                if "class Solution" in src or f"def {entry}" in src: return src
                break
            except SyntaxError: lines.pop()
    return (completion or "")[:CAP]

def clip(s):
    s = s or ""
    return s if len(s) <= CAP else s[:CAP] + "\n…(truncated)"

def load(p): return json.load(open(os.path.join(ROOT, p), encoding="utf-8"))

# ---------- styling ----------
HEAD_FILL = PatternFill("solid", fgColor="1F3864")
HEAD_FONT = Font(name="Arial", bold=True, color="FFFFFF", size=11)
PASS_FILL = PatternFill("solid", fgColor="C6EFCE")
FAIL_FILL = PatternFill("solid", fgColor="FFC7CE")
PASS_FONT = Font(name="Arial", color="006100", bold=True)
FAIL_FONT = Font(name="Arial", color="9C0006", bold=True)
BASE_FONT = Font(name="Arial", size=10)
MONO = Font(name="Consolas", size=9)
THIN = Border(*[Side(style="thin", color="D9D9D9")]*4)
TOP = Alignment(vertical="top", wrap_text=True)

def style_header(ws, ncols):
    for c in range(1, ncols+1):
        cell = ws.cell(1, c); cell.fill = HEAD_FILL; cell.font = HEAD_FONT
        cell.alignment = Alignment(vertical="center", horizontal="left")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(ncols)}{ws.max_row}"
    ws.row_dimensions[1].height = 22

def write_rows(ws, headers, rows, widths, result_cols=()):
    ws.append(headers)
    for r in rows:
        ws.append(r)
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    # body font + borders + wrap
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = BASE_FONT; cell.alignment = TOP; cell.border = THIN
    # code columns monospace (last col assumed code if wide)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if ws.column_dimensions[cell.column_letter].width >= 60:
                cell.font = MONO; cell.alignment = TOP
    # color PASS/FAIL cells
    for ci in result_cols:
        for row in ws.iter_rows(min_row=2, min_col=ci, max_col=ci):
            for cell in row:
                v = str(cell.value)
                if v == "PASS": cell.fill = PASS_FILL; cell.font = PASS_FONT
                elif v == "FAIL": cell.fill = FAIL_FILL; cell.font = FAIL_FONT
    style_header(ws, len(headers))

wb = openpyxl.Workbook()

# ============ README ============
ws = wb.active; ws.title = "README"
readme = [
 ("The Vibe Tax — Problem-by-Problem Breakdown", 16, True),
 ("", 10, False),
 ("Per-problem, per-model success/failure across three benchmarks. One tab per benchmark.", 11, False),
 ("Framing is null (terse/casual/polite/multilingual do not differ), so each row shows the", 11, False),
 ("TERSE condition as representative; the LCB tabs also give a pass-count across all 4 framings.", 11, False),
 ("", 10, False),
 ("Tabs", 13, True),
 ("• LCB — one row per (problem × model). difficulty, topic, the problem, PASS/FAIL, the model's code.", 11, False),
 ("• LCB by problem — one row per problem: how many of the 12 attempts (4 framings × 3 models) solved it.", 11, False),
 ("• HumanEval — HE4 set: the SAME 4 framings as LCB, base HumanEval tests (saturated; ~all pass). Model code + passed /4.", 11, False),
 ("• HumanEval+ — v3 set, EvalPlus edge-case tests (harder). base vs plus pass shown. (HE4 has no edge-test scoring.)", 11, False),
 ("• GPT-5.6 Failures — every problem×framing the GPT-5.6 ablation failed (148), flagged REGRESSION where GPT-5.4 had passed.", 11, False),
 ("", 10, False),
 ("Key columns", 13, True),
 ("• result / PASS-FAIL: did the model's code pass the graded tests for that problem.", 11, False),
 ("• model_solution: the actual code the model produced (robustly extracted; this is what was run).", 11, False),
 ("• topic (LCB): keyword-derived, directional not gold-standard (LCB ships no topic labels).", 11, False),
 ("", 10, False),
 ("What to look for", 13, True),
 ("• LCB failures are 100% wrong-logic: valid code, wrong answer — the model can code, not solve.", 11, False),
 ("• Difficulty drives success (capable models): easy 98.5%, medium 87.3%, hard 60.5%. Dynamic programming is the weakest topic.", 11, False),
 ("• HumanEval is saturated (~97%) — few failures to see there; LCB is where the signal is.", 11, False),
 ("", 10, False),
 ("Sources: LCB tabs = data/vibe_tax_lcb/lcb_scored.json + lcb_v3_responses.json + lcb_problems.jsonl;", 9, False),
 ("HumanEval tab = data/vibe_tax_v2/he4_scored.json + he4_responses.json (HE4, 4 framings);", 9, False),
 ("HumanEval+ tab = data/vibe_tax_v3/vibe_tax_v3_plus_scored.json + _responses.json. Numbers are pass@1, temp 0.", 9, False),
]
for i,(txt,sz,bold) in enumerate(readme,1):
    c = ws.cell(i,1,txt); c.font = Font(name="Arial", size=sz, bold=bold)
ws.column_dimensions["A"].width = 110

# ============ LCB ============
lcb_scored = load("data/vibe_tax_lcb/lcb_scored.json")
lcb_resp = {(x["task_id"],x["level"],x["model"]): x for x in load("data/vibe_tax_lcb/lcb_v3_responses.json")}
probs = {json.loads(l)["task_id"]: json.loads(l) for l in open(os.path.join(ROOT,"data/vibe_tax_lcb/lcb_problems.jsonl"))}
for p in probs.values(): p["topic"] = classify(p["question_content"]+" "+p["entry_point"])

# passed lookup and per-(problem,model) count across framings
passed = {(x["task_id"],x["level"],x["model"]): x["passed"] for x in lcb_scored}
from collections import defaultdict
cnt = defaultdict(lambda:[0,0])
for x in lcb_scored:
    cnt[(x["task_id"],x["model"])][0]+=x["passed"]; cnt[(x["task_id"],x["model"])][1]+=1

rows=[]
for tid in sorted(probs):
    p=probs[tid]
    for model in ("chatgpt","claude","codestral"):
        pv = passed.get((tid,REP,model))
        if pv is None: continue
        k,n = cnt[(tid,model)]
        comp = lcb_resp.get((tid,REP,model),{}).get("completion")
        rows.append([tid, p["difficulty"], p["topic"], p["entry_point"], model,
                     "PASS" if pv else "FAIL", f"{k}/{n}",
                     clip(p["question_content"]), lcb_code(comp,p["entry_point"])])
# sort: hardest + most-failed first for quick insight
order={"hard":0,"medium":1,"easy":2}
rows.sort(key=lambda r:(order.get(r[1],3), r[5]!="FAIL", r[0], r[4]))
ws = wb.create_sheet("LCB")
write_rows(ws,
    ["task_id","difficulty","topic","method","model","result (terse)","passed /4 (all framings)","problem_statement","model_solution"],
    rows, [11,9,18,26,10,13,20,70,80], result_cols=(6,))

# ============ LCB by problem ============
rows=[]
for tid in sorted(probs):
    p=probs[tid]
    per={m:cnt.get((tid,m),[0,0]) for m in ("chatgpt","claude","codestral")}
    tot=sum(v[0] for v in per.values()); den=sum(v[1] for v in per.values())
    verdict = "Always" if tot==den and den>0 else ("Never" if tot==0 else "Partial")
    rows.append([tid,p["difficulty"],p["topic"],p["entry_point"],
                 f'{per["chatgpt"][0]}/{per["chatgpt"][1]}',
                 f'{per["claude"][0]}/{per["claude"][1]}',
                 f'{per["codestral"][0]}/{per["codestral"][1]}',
                 f"{tot}/{den}", verdict])
rows.sort(key=lambda r:(order.get(r[1],3), int(r[7].split("/")[0])))
ws = wb.create_sheet("LCB by problem")
write_rows(ws,
    ["task_id","difficulty","topic","method","chatgpt /4","claude /4","codestral /4","total /12","verdict"],
    rows, [11,9,18,30,11,11,12,10,10])
# color verdict
for row in ws.iter_rows(min_row=2, min_col=9, max_col=9):
    for cell in row:
        if cell.value=="Always": cell.fill=PASS_FILL; cell.font=PASS_FONT
        elif cell.value=="Never": cell.fill=FAIL_FILL; cell.font=FAIL_FONT

# ============ HumanEval (HE4, 4-condition matched set) ============
he_probs={}
for l in open(os.path.join(ROOT,"data/HumanEval.jsonl/human-eval-v2-20210705.jsonl")):
    d=json.loads(l); he_probs[d["task_id"]]=d

# HE4: the same 4 deterministic wrappers as LCB (terse/casual/detailed/multilingual)
he4_resp={(x["task_id"],x["level"],x["model"]):x for x in load("data/vibe_tax_v2/he4_responses.json")}
he4_pass={(x["task_id"],x["level"],x["model"]):x["passed"] for x in load("data/vibe_tax_v2/he4_scored.json")}
he4_cnt=defaultdict(lambda:[0,0])
for (t,l,m),p in he4_pass.items():
    he4_cnt[(t,m)][0]+=int(bool(p)); he4_cnt[(t,m)][1]+=1

rows=[]
for tid,model in sorted({(t,m) for (t,l,m) in he4_pass if l==REP}):
    pv=he4_pass.get((tid,REP,model))
    if pv is None: continue
    k,n=he4_cnt[(tid,model)]
    comp=he4_resp.get((tid,REP,model),{}).get("completion")
    ep=he_probs.get(tid,{}).get("entry_point","")
    rows.append([tid,ep,model,"PASS" if pv else "FAIL",f"{k}/{n}",
                 clip(he_probs.get(tid,{}).get("prompt","")),clip(comp)])
rows.sort(key=lambda r:(r[3]!="FAIL", r[0], r[2]))
ws=wb.create_sheet("HumanEval")
write_rows(ws, ["task_id","entry_point","model","result (terse)","passed /4 (all framings)","problem (spec)","model_solution"],
           rows, [12,26,10,13,20,70,80], result_cols=(4,))

# ============ HumanEval+ (v3 dataset — HE4 has no edge-test scoring) ============
v3_resp={(x["task_id"],x["level"],x["model"]):x for x in load("data/vibe_tax_v3/vibe_tax_v3_responses.json")}
v3_base={(x["task_id"],x["level"],x["model"]):x["passed"] for x in load("data/vibe_tax_v3/vibe_tax_v3_scored.json")}
v3_plus={(x["task_id"],x["level"],x["model"]):x["passed"] for x in load("data/vibe_tax_v3/vibe_tax_v3_plus_scored.json")}

ws=wb.create_sheet("HumanEval+")
rows=[]
for tid,model in sorted({(t,m) for (t,l,m) in v3_resp if l==REP}):
    b=v3_base.get((tid,REP,model)); pl=v3_plus.get((tid,REP,model))
    if pl is None: continue
    comp=v3_resp.get((tid,REP,model),{}).get("completion"); ep=he_probs.get(tid,{}).get("entry_point","")
    rows.append([tid,ep,model,"PASS" if b else "FAIL","PASS" if pl else "FAIL",
                 clip(he_probs.get(tid,{}).get("prompt","")),clip(comp)])
rows.sort(key=lambda r:(r[4]!="FAIL", r[0], r[2]))
write_rows(ws, ["task_id","entry_point","model","base result","plus result (edge tests)","problem (spec)","model_solution"],
           rows, [12,26,10,12,16,70,80], result_cols=(4,5))

# ============ GPT-5.6 Failures (ablation) ============
g56_path = os.path.join(ROOT, "data/vibe_tax_lcb/lcb_v3_gpt56_scored.json")
if os.path.exists(g56_path):
    g56 = json.load(open(g56_path, encoding="utf-8"))
    base54 = {(x["task_id"], x["level"]): x["passed"]
              for x in lcb_scored if x.get("model") == "chatgpt"}
    model_id = next((x.get("model_id") for x in g56), "gpt-5.6")
    rows = []
    for x in g56:
        if x["passed"]:
            continue
        tid = x["task_id"]; p = probs.get(tid, {})
        was_ok_54 = base54.get((tid, x["level"]))
        flag = "REGRESSION (5.4 passed)" if was_ok_54 is True else "also failed by 5.4"
        rows.append([tid, x["difficulty"], p.get("topic", ""), p.get("entry_point", ""),
                     x["level"], "FAIL", flag, clip(p.get("question_content", ""))])
    order = {"hard": 0, "medium": 1, "easy": 2}
    rows.sort(key=lambda r: (order.get(r[1], 3), r[6] != "REGRESSION (5.4 passed)", r[0], r[4]))
    ws = wb.create_sheet("GPT-5.6 Failures")
    write_rows(ws,
        [f"task_id", "difficulty", "topic", "method", "condition (framing)",
         f"{model_id} result", "vs GPT-5.4", "problem_statement"],
        rows, [11, 9, 18, 26, 22, 14, 24, 80], result_cols=(6,))
    # color the regression flag
    for row in ws.iter_rows(min_row=2, min_col=7, max_col=7):
        for cell in row:
            if str(cell.value).startswith("REGRESSION"):
                cell.fill = FAIL_FILL; cell.font = FAIL_FONT
    print(f"GPT-5.6 Failures tab: {len(rows)} failing cells "
          f"({sum(1 for r in rows if r[6].startswith('REGRESSION'))} regressions)")

# ============ Partial Credit (per-task, line-by-line) ============
# Per-ATTEMPT test-pass rate for EVERY LCB problem, over the 12 attempts
# (3 models x 4 framings). Source: data/vibe_tax_lcb/full_partial_credit.json.
#   n_solved  attempts passing ALL tests   best %  best single attempt
#   mean %    avg test-pass over 12         worst % weakest attempt
# A passed attempt = 100% (no re-run); only FAILED attempts are executed for
# their partial count. This lets you inspect capability as a continuum per task.
pc = load("data/vibe_tax_lcb/full_partial_credit.json")
by_tid = {x["task_id"]: x for x in pc}

def pc_status(x):
    if x["n_solved"] == x["n_attempts"]:
        return "fully_solved"
    if x["n_solved"] == 0:
        return "never_solved"
    return "partially_solved"

rows = []
for tid in sorted(by_tid):
    x = by_tid[tid]
    p = probs.get(tid, {})
    rows.append([tid, x["difficulty"], x["topic"], p.get("entry_point", ""),
                 pc_status(x), x["n_solved"], x["n_attempts"],
                 x["best"], x["mean"], x["worst"], clip(p.get("question_content", ""))])
# difficulty groups, then most-partial (lowest mean) first within each group
order = {"easy": 0, "medium": 1, "hard": 2}
rows.sort(key=lambda r: (order.get(r[1], 3), r[8]))

ws = wb.create_sheet("Partial Credit")
# --- summary header computed from the per-task rows themselves ---
from collections import defaultdict as _dd
bd = _dd(list)
for x in pc:
    bd[x["difficulty"]].append(x)
fully = sum(1 for x in pc if x["n_solved"] == x["n_attempts"])
part = sum(1 for x in pc if 0 < x["n_solved"] < x["n_attempts"])
never = sum(1 for x in pc if x["n_solved"] == 0)
hdr = [
    ("Partial Credit — capability is a continuum (per-attempt TEST-pass rate, all 167 LCB problems)", 13, True),
    ("Each problem gets 12 attempts (3 models x 4 framings). mean = avg test-pass % over all 12; best = best single attempt.", 10, False),
    ("A passed attempt counts as 100%; only failed attempts are re-run for their partial test count.", 10, False),
    ("", 10, False),
    ("difficulty   mean-attempt   best-attempt   fully-solved", 10, True),
]
for d in ("easy", "medium", "hard"):
    g = bd[d]
    if g:
        mean = round(sum(x["mean"] for x in g) / len(g), 1)
        best = round(sum(x["best"] for x in g) / len(g), 1)
        fs = sum(1 for x in g if x["n_solved"] == x["n_attempts"])
        hdr.append((f"{d:8}       {mean:5}%         {best:5}%          {fs}/{len(g)}", 10, False))
hdr += [
    ("", 10, False),
    (f"Totals:  fully solved {fully}  |  partially solved {part}  |  never solved {never}   (of 167)", 10, True),
    ("NOTE: this per-task pass was run at a 2s-per-test cap (see CAPABILITY_ANALYSIS.md §4b), so it lists "
     f"{never} never-solved; the authoritative problem-level re-score (6s cap, 'LCB by problem' tab) finds 10 — "
     "the extras are correct-but-slow solutions. Re-run `full_partial_credit.py --timeout 6` to align.", 8, False),
    ("", 10, False),
]
for ri, (txt, sz, bold) in enumerate(hdr, 1):
    c = ws.cell(ri, 1, txt)
    c.font = Font(name="Arial", size=sz, bold=bold, italic=(sz == 8))
# --- per-task table below the header ---
table_start = len(hdr) + 1
headers = ["task_id", "difficulty", "topic", "method", "status",
           "n_solved", "n_attempts", "best %", "mean %", "worst %", "problem_statement"]
ws.append(headers)
for r in rows:
    ws.append(r)
widths = [11, 9, 20, 26, 16, 9, 11, 8, 8, 8, 80]
for i, w in enumerate(widths, 1):
    ws.column_dimensions[get_column_letter(i)].width = w
for row in ws.iter_rows(min_row=table_start + 1):
    for cell in row:
        cell.font = BASE_FONT; cell.alignment = TOP; cell.border = THIN
        if cell.column_letter == "K":
            cell.font = MONO
# color the status column
for row in ws.iter_rows(min_row=table_start + 1, min_col=5, max_col=5):
    for cell in row:
        if cell.value == "fully_solved":
            cell.fill = PASS_FILL; cell.font = PASS_FONT
        elif cell.value == "never_solved":
            cell.fill = FAIL_FILL; cell.font = FAIL_FONT
# style + freeze the table header row
for c in range(1, len(headers) + 1):
    cell = ws.cell(table_start, c); cell.fill = HEAD_FILL; cell.font = HEAD_FONT
    cell.alignment = Alignment(vertical="center", horizontal="left")
ws.freeze_panes = ws.cell(table_start + 1, 1)
ws.auto_filter.ref = f"A{table_start}:{get_column_letter(len(headers))}{ws.max_row}"
print(f"Partial Credit tab: {len(rows)} problems "
      f"(fully {fully} / partial {part} / never {never})")

wb.save(OUT)
print("wrote", OUT)
for s in wb.sheetnames: print("  tab:", s, "rows:", wb[s].max_row-1 if s!="README" else "-")
