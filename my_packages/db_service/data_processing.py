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



