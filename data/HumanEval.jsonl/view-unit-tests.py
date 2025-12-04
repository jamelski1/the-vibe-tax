import json

with open('human-eval-v2-20210705.jsonl', 'r', encoding='utf-8') as f:
    with open('all_tests.txt', 'w', encoding='utf-8') as out:
        for line in f:
            problem = json.loads(line)
            out.write(f"\n{'='*80}\n")
            out.write(f"{problem['task_id']}\n")
            out.write(f"{'='*80}\n")
            out.write("FUNCTION SIGNATURE:\n")
            out.write(problem['prompt'])
            out.write("\n\nUNIT TESTS:\n")
            out.write(problem['test'])
            out.write("\n\n")

print("All tests extracted to all_tests.txt")