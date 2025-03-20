import json
import os
from turtle import pd
from matplotlib import pyplot as plt
from my_packages.db_service import db, project_root

### 📌 EXPORT DATA FROM MONGODB ###
def export_collection(experiment: str, collection_type: str, file_format="json", exclude_columns=[]):
    """
    Exports a MongoDB collection to JSON or CSV.

    Parameters:
        - experiment (str): The name of the experiment.
        - collection_type (str): The type of collection to export ('errors', 'results', 'best_params').
        - file_format (str): The output format ("json" or "csv"). Default is "json".
    """

    EXPORT_DIR = os.path.join(project_root, "experiments/few-shot/db_exports")
    os.makedirs(EXPORT_DIR, exist_ok=True)

    collection_name = f"{experiment}_{collection_type}"
    collection = db[collection_name]

    # Prepare projection dictionary to exclude columns
    projection = {"_id": 0}  # Always exclude MongoDB's `_id`
    if exclude_columns:
        for col in exclude_columns:
            projection[col] = 0  # Set to 0 to exclude the column

    data = list(collection.find({}, projection))  # Exclude MongoDB ObjectId

    if not data:
        print(f"⚠️ No data found in collection '{collection_name}' for experiment '{experiment}'.")
        return

    # Define file paths
    export_file = os.path.join(EXPORT_DIR, f"{experiment}_{collection_type}.{file_format}")

    # Export as JSON
    if file_format.lower() == "json":
        with open(export_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"✅ Data exported to {export_file}")

    # Export as CSV
    elif file_format.lower() == "csv":
        df = pd.DataFrame(data)
        df.to_csv(export_file, index=False)
        print(f"✅ Data exported to {export_file}")

    else:
        print(f"❌ Invalid format '{file_format}'. Choose 'json' or 'csv'.")


### 📌 EXPORT ALL COLLECTIONS ###
def export_all_collections(experiment: str, file_format="json"):
    """
    Exports all collections (errors, results, best_params) for an experiment.

    Parameters:
        - experiment (str): The name of the experiment.
        - file_format (str): The output format ("json" or "csv"). Default is "json".
    """
    collections = ["errors", "results", "best_params"]
    for collection in collections:
        export_collection(experiment, collection, file_format)
