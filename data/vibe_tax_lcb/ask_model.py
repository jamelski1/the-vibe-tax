"""
Send ONE LiveCodeBench prompt to ONE model via API and save the reply to a file,
so you can test it with manual_test.py — no browser copy-paste.

    python ask_model.py <task_id> <condition> <model> [outfile]

    python ask_model.py lcb/3550 casual chatgpt              # -> reply.txt
    python ask_model.py lcb/3550 terse  claude  r_claude.txt

  condition : terse | casual | detailed | multilingual  (or the full names)
  model     : chatgpt | claude | codestral   (needs that provider's API key)

Uses the SAME system prompt, wrappers, temperature and max-tokens as the real
run — controlled by the same env vars, e.g. for a reasoning model:
    $env:OPENAI_MODEL="gpt-5.6"; $env:API_TEMPERATURE="none"; $env:MAX_TOKENS="16000"
The exact model_id and temperature used are printed so you know what you tested.

Then:  python manual_test.py <task_id> <outfile>
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)                                   # print_prompt
sys.path.insert(0, os.path.join(ROOT, "data", "vibe_tax_v2"))   # run_vibe_tax

from print_prompt import WRAPPERS, ALIASES, load                # noqa: E402
import run_vibe_tax as R                                        # noqa: E402  (query/make_clients/MODEL_ID/TEMPERATURE/MAX_TOKENS)


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: python ask_model.py <task_id> <condition> <model> [outfile]\n"
                 "  condition: terse|casual|detailed|multilingual   model: chatgpt|claude|codestral")
    task_id, condition, model = sys.argv[1], sys.argv[2].lower(), sys.argv[3].lower()
    outfile = sys.argv[4] if len(sys.argv) > 4 else "reply.txt"

    probs = load()
    if task_id not in probs:
        sys.exit(f"{task_id} not found (try: python print_prompt.py --list)")
    cond = ALIASES.get(condition, condition)
    if cond not in WRAPPERS:
        sys.exit(f"unknown condition '{condition}' (choose {list(WRAPPERS)})")

    p = probs[task_id]
    cls = p.get("class_name") or "Solution"
    user = WRAPPERS[cond](cls, p["entry_point"], p["params"], p["question_content"])

    clients = R.make_clients()
    if model not in clients:
        sys.exit(f"model '{model}' unavailable — set its API key "
                 f"(OPENAI_API_KEY / ANTHROPIC_API_KEY / CODESTRAL_API_KEY). "
                 f"available now: {list(clients)}")
    api_type, client = clients[model]

    mid = R.MODEL_ID[api_type]
    print(f"asking {model} ({mid})  temp={R.TEMPERATURE}  max_tokens={R.MAX_TOKENS}")
    print(f"  {task_id} | {p['difficulty']} | condition={cond} ...")
    completion = R.query(api_type, client, user)
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(completion or "")
    n = len(completion or "")
    print(f"saved {n} chars -> {outfile}"
          + ("   ⚠️ EMPTY reply (reasoning model may need a higher MAX_TOKENS)" if n == 0 else ""))
    print(f"now run:  python manual_test.py {task_id} {outfile}")


if __name__ == "__main__":
    main()
