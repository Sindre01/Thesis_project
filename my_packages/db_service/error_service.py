### 📌 SAVE ERRORS TO MONGODB ###
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from my_packages.common.classes import CodeEvaluationResult
from my_packages.db_service import db

def save_errors_to_db(
        experiment: str,
        model_name: str, 
        test_results: dict[int, list[CodeEvaluationResult]], 
        hyperparams: dict, 
        phase: str,
        eval_method: str = "hold_out",
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
        
def errors_to_df(experiment: str, model: str = None) -> pd.DataFrame:
    """Loads a MongoDB collection into a Pandas DataFrame."""
    collection_name = f"{experiment}_errors"
    collection = db[collection_name]
    if model:
        data = list(collection.find({"model_name": model}, {"_id": 0}))
    else:
        data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB ObjectId

    if not data:
        raise ValueError(f"Collection '{collection_name}' is empty in MongoDB.")

    return pd.DataFrame(data)

def pretty_print_errors(
        experiment: str, 
        filter= {}, 
        limit=10,
        exclude_columns=["model_name", "candidate_id", "phase", "seed", "temperature",  "top_k", "top_p", "created_at"]
        ):
    """Pretty prints the errors for an experiment."""
    collection_name = f"{experiment}_errors"
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
    if "metric" in df.columns:
        df = df.sort_values(by=["metric"], ascending=[True])

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
    if collection_name == f"{experiment}_errors":
        extra_info = f"validation: {collection.count_documents({'phase': 'validation'})}, testing: {collection.count_documents({'phase': 'testing'})}"


    print(f"Total documents/rows: {collection.count_documents({})}      {extra_info}")
    print("-" * 50)

def make_error_dataset(phase: str, experiment: str = None, output_file: str = None) -> pd.DataFrame:
    """Creates a dataset of errors for all models in all experiemnts"""
    errors = []
    if experiment:
        errors.append(errors_to_df(experiment))
    else:
        for collection in list_errors_collections():
            errors.append(errors_to_df(collection))

    return pd.concat(errors)
#     df_errors = errors_to_df(experiment, model)
#     if df_errors.empty:
#         print(f"⚠️ No errors found for experiment '{experiment}'.")
#         return df_errors

#     # Drop unnecessary columns
#     df_errors.drop(columns=["code_candidate", "stderr", "stdout"], inplace=True)

#     # Rename columns
#     df_errors.rename(columns={
#         "model_name": "Model",
#         "task_id": "Task ID",
#         "candidate_id": "Candidate ID",
#         "metric": "Metric",
#         "error_type": "Error Type",
#         "error_msg": "Error Message",
#         "test_result": "Test Result",
#         "phase": "Phase",
#         "seed": "Seed",
#         "temperature": "Temperature",
#         "top_p": "Top P",
#         "top_k": "Top K",
#         "created_at": "Created At"
#     }, inplace=True)

#     return df_errors

