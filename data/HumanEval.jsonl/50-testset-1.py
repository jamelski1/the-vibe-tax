import json

# Define the 50 medium-difficulty problems by category
medium_problems = {
    "String Manipulation & Pattern Matching": [1, 6, 10, 22, 28, 41, 43, 58, 67, 153],
    "List Processing & Transformations": [0, 4, 9, 18, 25, 33, 34, 35, 54, 68],
    "Mathematical & Numerical": [13, 30, 32, 37, 48, 49, 60, 63, 66, 157],
    "Logic & Conditionals": [24, 36, 39, 40, 42, 55, 64, 75, 77, 80],
    "Data Structures & Comparison": [52, 59, 65, 71, 78, 96, 152, 155, 158, 163]
}

# Load all problems into a dictionary for easy lookup
problems_dict = {}
with open('human-eval-v2-20210705.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        problem = json.loads(line)
        # Extract problem number from task_id (e.g., "HumanEval/0" -> 0)
        problem_num = int(problem['task_id'].split('/')[-1])
        problems_dict[problem_num] = problem

# Write organized output
with open('medium_difficulty_problems.txt', 'w', encoding='utf-8') as out:
    out.write("="*80 + "\n")
    out.write("HUMANEVAL MEDIUM-DIFFICULTY PROBLEMS (50 Selected)\n")
    out.write("="*80 + "\n\n")
    
    for category, problem_nums in medium_problems.items():
        out.write("\n" + "#"*80 + "\n")
        out.write(f"# {category.upper()}\n")
        out.write("#"*80 + "\n\n")
        
        for num in problem_nums:
            problem = problems_dict[num]
            out.write(f"\n{'='*80}\n")
            out.write(f"{problem['task_id']}\n")
            out.write(f"{'='*80}\n")
            
            out.write("\n--- FUNCTION SIGNATURE & DOCSTRING ---\n")
            out.write(problem['prompt'])
            
            out.write("\n\n--- UNIT TESTS ---\n")
            out.write(problem['test'])
            
            out.write(f"\n\n--- ENTRY POINT ---\n")
            out.write(f"Function name: {problem['entry_point']}\n")
            
            out.write("\n")

print("Medium-difficulty problems saved to medium_difficulty_problems.txt")
print(f"Total problems extracted: {sum(len(nums) for nums in medium_problems.values())}")