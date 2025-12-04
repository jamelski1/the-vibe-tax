import json
import time
from anthropic import Anthropic

# Initialize the Anthropic client
client = Anthropic()

# Define the 50 medium-difficulty problems by category
medium_problems = {
    "String Manipulation & Pattern Matching": [1, 6, 10, 22, 28, 41, 43, 58, 67, 153],
    "List Processing & Transformations": [0, 4, 9, 18, 25, 33, 34, 35, 54, 68],
    "Mathematical & Numerical": [13, 30, 32, 37, 48, 49, 60, 63, 66, 157],
    "Logic & Conditionals": [24, 36, 39, 40, 42, 55, 64, 75, 77, 80],
    "Data Structures & Comparison": [52, 59, 65, 71, 78, 96, 152, 155, 158, 163]
}

# Flatten to get all problem numbers
all_medium_problems = [num for nums in medium_problems.values() for num in nums]

# Load all problems
problems_dict = {}
with open('human-eval-v2-20210705.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        problem = json.loads(line)
        problem_num = int(problem['task_id'].split('/')[-1])
        problems_dict[problem_num] = problem

# System prompt for generating informal versions
def get_system_prompt(formality_level):
    prompts = {
        2: "You are converting formal programming docstrings into slightly less formal versions. Maintain technical accuracy but use a more conversational tone. Keep all specifications and examples intact.",
        3: "You are converting formal programming docstrings into casual versions. Use everyday language while preserving all technical requirements. Think of explaining to a colleague over coffee.",
        4: "You are converting formal programming docstrings into very casual versions. Use informal language, contractions, and a friendly tone. Imagine texting a friend who codes.",
        5: "You are converting formal programming docstrings into extremely casual, 'vibes-based' versions. Use slang, be vague about details, focus on the general idea. Think of how someone would describe what they need without reading documentation."
    }
    return prompts[formality_level]

def generate_informal_prompt(docstring, formality_level):
    """Generate an informal version of the docstring at the specified formality level."""
    
    system_prompt = get_system_prompt(formality_level)
    
    user_prompt = f"""Convert this formal programming docstring to formality level {formality_level}:

{docstring}

Provide ONLY the converted docstring, no explanations or preamble."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",  # Best Claude model currently available
        max_tokens=1000,
        temperature=0,  # Zero temperature for consistency
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return message.content[0].text

# Generate the vibe spectrum
output_data = []

print("Generating vibe spectrum for 50 medium-difficulty problems...")
print("This will make 200 API calls (50 problems × 4 informal levels)")
print("Estimated time: ~10-15 minutes\n")

for i, problem_num in enumerate(all_medium_problems, 1):
    problem = problems_dict[problem_num]
    task_id = problem['task_id']
    original_prompt = problem['prompt']
    
    print(f"Processing {i}/50: {task_id}")
    
    # Level 1 (Formal) - Original docstring
    spectrum_entry = {
        "task_id": task_id,
        "problem_number": problem_num,
        "entry_point": problem['entry_point'],
        "level_1_formal": original_prompt,
        "level_2": None,
        "level_3": None,
        "level_4": None,
        "level_5_vibe": None
    }
    
    # Generate levels 2-5
    for level in range(2, 6):
        try:
            informal_version = generate_informal_prompt(original_prompt, level)
            spectrum_entry[f"level_{level}" if level < 5 else "level_5_vibe"] = informal_version
            print(f"  ✓ Level {level} generated")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ✗ Error generating level {level}: {e}")
            spectrum_entry[f"level_{level}" if level < 5 else "level_5_vibe"] = f"ERROR: {str(e)}"
    
    output_data.append(spectrum_entry)
    print()

# Save to JSON file
with open('vibe_spectrum_data.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("✓ Complete! Data saved to vibe_spectrum_data.json")

# Also create a human-readable text version
with open('vibe_spectrum_readable.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("VIBE SPECTRUM: MEDIUM-DIFFICULTY HUMANEVAL PROBLEMS\n")
    f.write("="*80 + "\n\n")
    
    for entry in output_data:
        f.write(f"\n{'#'*80}\n")
        f.write(f"# {entry['task_id']}\n")
        f.write(f"{'#'*80}\n\n")
        
        for level in range(1, 6):
            level_key = f"level_{level}_formal" if level == 1 else (f"level_{level}_vibe" if level == 5 else f"level_{level}")
            f.write(f"\n--- LEVEL {level} {'(FORMAL - Original Docstring)' if level == 1 else '(VIBE - Extremely Casual)' if level == 5 else ''} ---\n")
            f.write(entry[level_key] or "ERROR")
            f.write("\n")

print("✓ Human-readable version saved to vibe_spectrum_readable.txt")
print(f"\nTotal problems processed: {len(output_data)}")