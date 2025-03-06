from datetime import datetime
from zoneinfo import ZoneInfo
import re
import pandas as pd
from my_packages.common import Run
from my_packages.db_service.data_processing import flatten_metric_results
from my_packages.db_service import db

def save_results_to_db(
        experiment: str, 
        model_name: str, 
        seeds: list[int], 
        ks: list[int],
        metrics: list[str],
        result: Run,
        db_connection=None
    ):
    """Saves results to MongoDB in the '{experiment}_results' collection."""
    # Use the provided connection or fall back to the global one
    if db_connection is None:
        db_connection = db
    collection = db_connection[f"{experiment}_results"]
    # Dynamically flatten the nested metric results.
    flattened_metrics = flatten_metric_results(result.metric_results)
    
    result_doc = {
        "model_name": model_name,
        "seed": seeds,
        "temperature": result.temperature,
        "top_p": result.top_p,
        "top_k": result.top_k,
        "ks": ks,
        "metrics": metrics,
        "created_at": datetime.now(ZoneInfo("Europe/Oslo")),
        **flattened_metrics,
    }

    collection.insert_one(result_doc)
    print(f"✅ Results saved to MongoDB for model '{model_name}' under experiment '{experiment}'.")
    return result_doc


def delete_results_collection(experiment: str):
    """
    Deletes results collection related to an experiment.

    """
    collection_name = f"{experiment}_results"

    if collection_name not in db.list_collection_names():
        print(f"⚠️ Collection '{collection_name}' does not exist in the database.")
        return

    confirmation = input(f"❗ Are you sure you want to delete collection '{collection_name}'? (yes/no): ").strip().lower()
    
    if confirmation == "yes":
        db[collection_name].drop()
        print(f"✅ Collection '{collection_name}' has been deleted successfully.")
    else:
        print("❌ Deletion cancelled.")


def list_results_collections() -> list[str]:
    """
    Lists all collections for results in the MongoDB database.
    """
    return [coll for coll in db.list_collection_names() if coll.endswith("_results")]
  
def results_to_df(experiment: str):
    """Loads a MongoDB collection into a Pandas DataFrame."""
    collection_name = f"{experiment}_results"
    collection = db[collection_name]
    
    data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB ObjectId

    if not data:
        raise ValueError(f"Collection '{collection_name}' is empty in MongoDB.")

    return pd.DataFrame(data)



import re
import os
import pandas as pd

import re
import os
import pandas as pd

def pretty_print_comparison_results(
        experiment: str,
        filter={},
        limit=10,
        exclude_columns=["candidate_id", "phase", "seed", "temperature",  "top_k", "top_p", "created_at"]
        ):
    """Pretty prints a comparison of results for 1-shot, 5-shot, and 10-shot experiments."""
    shot_levels = ["1_shot", "5_shot", "10_shot"]

    all_data = []
    for shot in shot_levels:
        collection_name = f"{experiment}_{shot}_results"
        collection = db[collection_name]

        projection = {"_id": 0}
        for col in exclude_columns:
            projection[col] = 0

        documents = list(collection.find(filter, projection).limit(limit))

        if not documents:
            print(f"⚠️ No data found in collection: {collection_name}")
            continue

        df = pd.DataFrame(documents)

        # Expand dictionary metrics into separate columns
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, dict)).any():
                expanded_cols = df[col].apply(pd.Series)
                expanded_cols = expanded_cols.add_prefix(f"{col}_")
                df = pd.concat([df.drop(columns=[col]), expanded_cols], axis=1)

        df["shot"] = shot
        all_data.append(df)

    if not all_data:
        print("⚠️ No data available for any shot level.")
        return

    combined_df = pd.concat(all_data)

    print("Available columns after expansion:", combined_df.columns.tolist())

    metric_columns = [col for col in combined_df.columns if any(metric in col for metric in ['syntax@', 'semantic@', 'tests@'])]
    combined_df = combined_df.pivot_table(index="model_name", columns="shot", values=metric_columns, aggfunc='first')
    combined_df.columns = [' '.join(col).strip() for col in combined_df.columns.values]
    combined_df.reset_index(inplace=True)

    combined_df = combined_df.sort_values(by="model_name")

    print(combined_df.to_string(index=False))
    print("...")

    for shot in shot_levels:
        collection_name = f"{experiment}_{shot}_results"
        collection = db[collection_name]
        count = collection.count_documents(filter)
        print(f"📊 {shot}: {count} documents")

    total_docs = sum(db[f"{experiment}_{shot}_results"].count_documents(filter) for shot in shot_levels)
    print(f"Total documents/rows across all shots: {total_docs}")
    print("-" * 50)

    return combined_df

    
def extract_size(model: str):
    if ":" not in model:
        return "N/A"
    return model.split(":")[1].strip().split("-")[0]