### 📌 SAVE ERRORS TO MONGODB ###
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from my_packages.common import CodeEvaluationResult
from my_packages.db_service import db

def save_errors_to_db(
        experiment: str,
        model_name: str, 
        test_results: dict[int, list[CodeEvaluationResult]], 
        hyperparams: dict, 
        phase: str,
        db_connection=None
    ):
    if db_connection is None:
        db_connection = db
    """Saves errors to MongoDB in the '{experiment}_errors' collection."""
    collection = db_connection[f"{experiment}_errors"]

    errors = []
    for task_id, results in test_results.items():
        for result in results:
            if not result.passed:  # Only log errors
                errors.append({
                    "model_name": model_name,
                    "task_id": task_id,
                    "candidate_id": result.candidate_id,
                    "metric": result.metric,
                    "error_type": result.error_type,
                    "error_msg": result.error_msg,
                    "code_candidate": result.code,
                    "test_result": result.test_result,
                    "stderr": result.compiler_msg.stderr if result.compiler_msg else "N/A",
                    "stdout": result.compiler_msg.stdout if result.compiler_msg else "N/A",
                    "phase": phase,
                    "seed": (hyperparams["seed"] * result.candidate_id),
                    "temperature": hyperparams["temperature"],
                    "top_p": hyperparams["top_p"],
                    "top_k": hyperparams["top_k"],
                    "created_at": datetime.now(ZoneInfo("Europe/Oslo"))

                })

    if errors:
        collection.insert_many(errors)
        # print(f"✅ Errors saved in MongoDB for model '{model_name}' under experiment '{experiment}'.")

def delete_errors_collection(experiment: str):
    """
    Deletes a specific collection related to an experiment.

    Parameters:
    - experiment (str): The name of the experiment.
    - collection_type (str): The type of collection to delete (e.g., 'errors', 'results', 'best_params').

    Example:
        delete_collection("signature_exp1", "results")
    """
    collection_name = f"{experiment}_errors"

    if collection_name not in db.list_collection_names():
        print(f"⚠️ Collection '{collection_name}' does not exist in the database.")
        return

    confirmation = input(f"❗ Are you sure you want to delete collection '{collection_name}'? (yes/no): ").strip().lower()
    
    if confirmation == "yes":
        db[collection_name].drop()
        print(f"✅ Collection '{collection_name}' has been deleted successfully.")
    else:
        print("❌ Deletion cancelled.")

def list_errors_collections() -> list[str]:
    """
    Lists all collections for errors in the MongoDB database.
    """
    return [coll for coll in db.list_collection_names() if coll.endswith("_errors")]  

def get_error_count_model(experiment: str, model_name: str, phase: str) -> int:
    """Returns the total count of errors for a model in an experiment."""
    errors_collection = db[f"{experiment}_errors"]
    return errors_collection.count_documents({"model_name": model_name, "phase": phase})
        
def errors_to_df(experiment: str):
    """Loads a MongoDB collection into a Pandas DataFrame."""
    collection_name = f"{experiment}_errors"
    collection = db[collection_name]
    
    data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB ObjectId

    if not data:
        raise ValueError(f"Collection '{collection_name}' is empty in MongoDB.")

    return pd.DataFrame(data)

