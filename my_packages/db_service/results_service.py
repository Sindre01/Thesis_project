from datetime import datetime
from zoneinfo import ZoneInfo
import re
import pandas as pd
from my_packages.common.classes import Run
from my_packages.db_service.data_processing import flatten_metric_results
from my_packages.db_service import db

def save_results_to_db(
        experiment: str, 
        model_name: str, 
        seeds: list[int], 
        ks: list[int],
        metrics: list[str],
        result: Run,
        db_connection=None,
        eval_method="hold_out"
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
        "eval_method": eval_method,
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
def get_db_results(
        experiment: str, 
        model:str, 
        eval_method: str
    ) -> list[dict]:
    """Checks if best parameters from validation exist in MongoDB."""
    collection = db[f"{experiment}_results"]
    results = collection.find_one({"model_name": model, "eval_method": eval_method})

    if results:
        results_dict = {
            f"pass@k_{metric}": [results[f"{metric}@{k}"] for k in results["ks"]]
            for metric in results["metrics"]
        }
        flattened_metrics = flatten_metric_results(results_dict)
        print(f"✅ Existing {eval_method} results for model '{model}' under experiment '{experiment}': {flattened_metrics}")
        return [{
            "model_name": results["model_name"],
            "metrics": results["metrics"],
            "seed": results["seed"],
            "temperature": results["temperature"],
            "top_p": results["top_p"],
            "top_k": results["top_k"],
            "ks": results["ks"],
            "eval_method": results["eval_method"],
            "created_at": results["created_at"].isoformat(),
            **flattened_metrics,
            }]
    
    print(f"⚠️ No best results found for model '{model}' under experiment '{experiment}'. Starting validation...")
    return None

def pretty_print_results(
        experiment: str, 
        filter= {}, 
        limit=10,
        exclude_columns=["seed", "ks", "created_at", "syntax@2"]
        ):
    """Pretty prints the results for an experiment."""
    collection_name = f"{experiment}_results"
    collection = db[collection_name]

    projection = {"_id": 0}  # Always exclude MongoDB's `_id`
    if exclude_columns:
        for col in exclude_columns:
            projection[col] = 0  # Set to 0 to exclude the column

    documents = list(collection.find(filter, projection).limit(limit))

    if not documents:
        print("⚠️ No data found in this collection.")
        return
    
    # Convert to DataFrame for better readability
    df = pd.DataFrame(documents)
    if "model_name" in df.columns:
        df["size"] = df["model_name"].apply(extract_size)  # Extract size
        df = df.sort_values(by=["size", "model_name"], ascending=[True, True])  # Sort by size first, then name
        df = df.drop(columns=["size"])  # Remove temporary column after sorting


    print(df.to_string(index=False))  # Pretty print DataFrame
    print("...")
    # Display document counts for each model_name
    model_counts = collection.aggregate([
        {"$group": {"_id": "$model_name", "count": {"$sum": 1}}}
    ])
    model_counts_dict = {entry["_id"]: entry["count"] for entry in model_counts}
    for model_name, count in model_counts_dict.items():
        print(f"📊 {model_name}: {count} documents")
    extra_info = ""

    print(f"Total documents/rows: {collection.count_documents({})}      {extra_info}")
    print("-" * 50)
    return df
    
def extract_size(model: str):
    if ":" not in model:
        return "N/A"
    return model.split(":")[1].strip().split("-")[0]