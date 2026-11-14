"""Paper name <-> code/data name, in one place.

The stored replies keep the keys they were collected under (`level`, `model`), so the raw
data is untouched; these tables translate between those keys and the names the paper uses.
"""

# Extractors: paper name -> key in extractors.EXTRACTORS (function is extract_<key>)
EXTRACTORS = {
    "Naive": "naive",
    "Sieve": "sieve",
    "sanitize": "sanitize",          # EvalPlus sanitize
    "LCB-official": "lcb_official",  # LiveCodeBench's own extract_code
}

# Registers: paper name -> `level` key stored on every reply
REGISTERS = {
    "terse": "agentic_terse",
    "casual": "agentic_casual",
    "multilingual": "webchat_multilingual",   # the Chinese-framing template
    "detailed": "webchat_detailed",           # the polite, detailed template
}
REGISTER_NAME = {v: k for k, v in REGISTERS.items()}   # stored key -> paper name

# Models: paper name -> `model` key stored on every reply (the paper's 1,336-reply slice)
MODELS = {
    "GPT-5.4": "chatgpt",
    "Claude Opus 4.6": "claude",
}
