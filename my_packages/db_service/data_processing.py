
# def refactor_pass_at_k_results(experiment: str):
#     """
#     Refactors the 'pass_at_k' nested fields in the results collection by flattening them
#     into individual top-level fields.

#     New fields created:
#       - syntax@1, syntax@2, syntax@5, syntax@10
#       - semantic@1, semantic@2, semantic@5, semantic@10
#       - tests@1, tests@2, tests@5, tests@10
#     """
#     results_collection = db[f"{experiment}_results"]
#     update_pipeline = [
#         {
#             "$set": {
#                 "syntax@1": "$pass_at_k.pass@k_syntax.pass@1",
#                 "syntax@2": "$pass_at_k.pass@k_syntax.pass@2",
#                 "syntax@5": "$pass_at_k.pass@k_syntax.pass@5",
#                 "syntax@10": "$pass_at_k.pass@k_syntax.pass@10",
#                 "semantic@1": "$pass_at_k.pass@k_semantic.pass@1",
#                 "semantic@2": "$pass_at_k.pass@k_semantic.pass@2",
#                 "semantic@5": "$pass_at_k.pass@k_semantic.pass@5",
#                 "semantic@10": "$pass_at_k.pass@k_semantic.pass@10",
#                 "tests@1": "$pass_at_k.pass@k_tests.pass@1",
#                 "tests@2": "$pass_at_k.pass@k_tests.pass@2",
#                 "tests@5": "$pass_at_k.pass@k_tests.pass@5",
#                 "tests@10": "$pass_at_k.pass@k_tests.pass@10"
#             }
#         }
#     ]
    
#     result = results_collection.update_many({}, update_pipeline)
#     print(f"Flattened fields in {result.modified_count} documents.")



def flatten_metric_results(metric_results: dict) -> dict:
    """
    Dynamically flattens the nested metric_results dictionary into individual key-value pairs.
    
    Expected input structure:
    {
        "pass@k_syntax": {
            "pass@1": {"mean": float, "std": float},
            "pass@2": {"mean": float, "std": float},
            "pass@5": {"mean": float, "std": float},
            "pass@10": {"mean": float, "std": float}
        },
        "pass@k_semantic": { ... },
        "pass@k_tests": { ... }
        // Possibly more keys...
    }
    
    For each top-level key, the function will:
      - Remove the "pass@k_" prefix (if present) to get the new prefix.
      - Iterate over the nested keys (e.g., "pass@1", "pass@2", etc.)
      - Create new keys of the form: <new_prefix>@<number>
    
    For example:
      "pass@k_syntax": {"pass@1": {...}, "pass@5": {...}}
    becomes:
      "syntax@1": {...}, "syntax@5": {...}
    """
    flattened = {}
    for original_field, nested in metric_results.items():
        # Dynamically derive a prefix by removing "pass@k_" if present.
        prefix = original_field.replace("pass@k_", "")
        # print("original field: ", original_field)
        # print("prefix: ", prefix)
        # Ensure that the nested value is a dictionary.
        if isinstance(nested, dict):
            for key, value in nested.items():
                # Extract the suffix from keys like "pass@1" (assumes this format).
                if "@" in key:
                    suffix = key.split("@")[1]
                else:
                    suffix = key
                new_key = f"{prefix}@{suffix}"
                flattened[new_key] = value
        else:
            # In case the nested value isn't a dictionary, just use the prefix as key.
            flattened[prefix] = nested

    return flattened



