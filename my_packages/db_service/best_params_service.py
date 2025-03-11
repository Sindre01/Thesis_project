from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from my_packages.db_service import db
from my_packages.common.classes import Run
from my_packages.db_service.data_processing import flatten_metric_results

def save_best_params_to_db(
    experiment: str, 
    model_name: str, 
    optimizer_metric: str, 
    best_params: Run, 
    db_connection=None,
    eval_method="hold_out"
):
    """Saves the best hyperparameters for a given model in MongoDB."""
    # Use the provided connection or fall back to the global one
    if db_connection is None:
        db_connection = db
    collection = db_connection[f"{experiment}_best_params"]

    # Remove old best params if they exist
    collection.delete_many({"model_name": model_name, "optimizer_metric": optimizer_metric})
    flattened_metrics = flatten_metric_results(best_params.metric_results)
    # Insert new best params
    collection.insert_one({
        "model_name": model_name,
        "optimizer_metric": optimizer_metric,
        "temperature": best_params.temperature,
        "top_p": best_params.top_p,
        "top_k": best_params.top_k,
        "seed": best_params.seed,
        "created_at": datetime.now(ZoneInfo("Europe/Oslo")),
        "eval_method": eval_method,
        **flattened_metrics
    })

    print(f"✅ Best parameters saved in MongoDB for model '{model_name}' under experiment '{experiment}'.")  

    ### 📌 CHECK IF BEST PARAMS EXIST ###
def get_db_best_params(
        experiment: str, 
        model:str, 
        metrics: list[str], 
        k: int,
        eval_method: str
    )-> list[dict]:
    """Checks if best parameters from validation exist in MongoDB."""
    collection = db[f"{experiment}_best_params"]
    results = []

    for optimizer_metric in metrics:
        best_params = collection.find_one({"model_name": model, "optimizer_metric": optimizer_metric, "eval_method": eval_method})

        if not best_params:
            print(f"⚠️ No best parameters '{optimizer_metric}' found for model '{model}' under experiment '{experiment}' on metric '{optimizer_metric}, on eval method '{eval_method}' .")
        else:
            results_dict = {
                f"pass@k_{metric}": [best_params[f"{metric}@{k}"] for k in best_params["ks"]]
                for metric in best_params["metrics"]
            }
            flattened_metrics = flatten_metric_results(results_dict)
            print(f"✅ Existing {eval_method} best parameters for model '{model}' optimized on metric '{optimizer_metric}@{k}': temperature={best_params['temperature']}, top_p={best_params['top_p']}, top_k={best_params['top_k']}, seed={best_params['seed']}")
            results.append({
                    "model_name": best_params["model_name"],
                    "optimizer_metric": optimizer_metric,
                    "seed": best_params["seed"],
                    "temperature": best_params["temperature"],
                    "top_p": best_params["top_p"],
                    "top_k": best_params["top_k"],
                    "eval_method": eval_method,
                    "created_at": best_params["created_at"].isoformat(),
                    **flattened_metrics,
                })
            
    return results
    
def delete_best_params_collection(experiment: str):
    """
    Deletes a specific collection related to an experiment.

    Parameters:
    - experiment (str): The name of the experiment.
    - collection_type (str): The type of collection to delete (e.g., 'errors', 'results', 'best_params').

    Example:
        delete_collection("signature_exp1", "results")
    """
    collection_name = f"{experiment}_best_params"

    if collection_name not in db.list_collection_names():
        print(f"⚠️ Collection '{collection_name}' does not exist in the database.")
        return

    confirmation = input(f"❗ Are you sure you want to delete collection '{collection_name}'? (yes/no): ").strip().lower()
    
    if confirmation == "yes":
        db[collection_name].drop()
        print(f"✅ Collection '{collection_name}' has been deleted successfully.")
    else:
        print("❌ Deletion cancelled.")


def list_params_collections() -> list[str]:
    """
    Lists all collections for best_params in the MongoDB database.
    """
    return [coll for coll in db.list_collection_names() if coll.endswith("_best_params")]

def best_params_to_df(experiment: str):
    """Loads a MongoDB collection into a Pandas DataFrame."""
    collection_name = f"{experiment}_best_params"
    collection = db[collection_name]
    
    data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB ObjectId

    if not data:
        raise ValueError(f"Collection '{collection_name}' is empty in MongoDB.")

    return pd.DataFrame(data)
