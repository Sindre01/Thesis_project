import json
import os
from collections import Counter

# Error category rankings
error_ranking = {
    "Tests": 3,     # Best
    "Semantic": 2,
    "Syntax": 1     # Worst
}
model="llama3.2:3b-instruct-fp16"
# Path to your folder containing JSON files
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = f"{script_dir}/RAG/signature/3_fold/signature_RAG_5_shot_{model}"  # <--- change this
print(f"Analyzing JSON files in: {folder_path}")


# Initialize counters
results_counter = Counter()

# Loop over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        print(f"Analyzing file: {filename}")
        
        with open(file_path, "r") as f:
            data = json.load(f)

        # If data is a list, iterate over each experiment
        experiments = data if isinstance(data, list) else [data]

        for experiment in experiments:
            task_candidates = experiment.get("task_candidates", [])
            
            for candidate in task_candidates:
                initial = candidate.get("initial_error_category")
                final = candidate.get("final_error_category")

                initial_rank = error_ranking.get(initial)
                final_rank = error_ranking.get(final)

                if initial_rank is None or final_rank is None:
                    print(f"Warning: Unknown error category in {filename}: {initial} or {final}")
                    continue

                if final_rank > initial_rank:
                    results_counter["better"] += 1
                elif final_rank < initial_rank:
                    results_counter["worse"] += 1
                else:
                    results_counter["same"] += 1

# Print the summary
print("\nError Category Change Summary across all JSON files:")
print(f"  Better: {results_counter['better']}")
print(f"  Worse:  {results_counter['worse']}")
print(f"  Same:   {results_counter['same']}")
