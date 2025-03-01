from datetime import datetime
from zoneinfo import ZoneInfo
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
