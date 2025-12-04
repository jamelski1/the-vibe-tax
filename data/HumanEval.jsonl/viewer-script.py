import json

with open('human-eval-v2-20210705.jsonl', 'r', encoding='utf-8') as f:
    with open('docstrings.txt', 'w', encoding='utf-8') as out:
        for line in f:
            problem = json.loads(line)
            out.write(f"\n{'='*80}\n")
            out.write(f"{problem['task_id']}\n")
            out.write(f"{'='*80}\n")
            out.write(problem['prompt'])
            out.write("\n\n")

print("Docstrings extracted to docstrings.txt")