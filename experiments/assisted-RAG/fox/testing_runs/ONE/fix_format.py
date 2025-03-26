import json

file = "./signature/3_fold/signature_RAG_10_shot_phi4:14b-fp16/fold_2.json"
with open(file) as f:
    data = json.load(f)

fixed_data = []

for entry in data:
    task_candidates = entry.get("task_candidates", {})
    largest_contexts = []

    # For each task id, get [candidates, context_value]
    for task_id, pair in task_candidates.items():
        if isinstance(pair, list) and len(pair) == 2 and isinstance(pair[1], int):
            task_candidates[task_id] = pair[0]  # Keep only the list of strings
            largest_contexts.append(pair[1])    # Collect context values

    # Store the max context at top level
    if largest_contexts:
        entry["largest_context"] = (max(largest_contexts) + 2012)

    fixed_data.append(entry)

# Save back to file
with open(file, "w") as f:
    json.dump(fixed_data, f, indent=2)
