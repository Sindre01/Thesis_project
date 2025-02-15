import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from my_packages.common import CodeEvaluationResult, Run

# Define project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# MongoDB Connection (Change URI if using a cloud database)
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)

# Define the main database for all experiments
DB_NAME = "few_shot_experiments"
db = client[DB_NAME]


def pretty_print_experiment_collections(experiment: str, limit=5, exclude_columns=["stderr", "stdout", "code_candidate"]):
    """
    Prints all collections related to an experiment in a readable format.

    Parameters:
    - experiment (str): The experiment name.
    - limit (int): Number of documents to show per collection (default: 5).
    - exclude_columns (list): List of column names to exclude from output.
    """
    print(f"\n🔍 Existing collections for {experiment}: ")
    print("=" * 50)

    # List all collections for the experiment
    collections = db.list_collection_names()
    filtered_collections = [col for col in collections if col.startswith(f"{experiment}_")]

    if not filtered_collections:
        print(f"⚠️ No collections found for experiment '{experiment}'.")
        return

    # Prepare projection dictionary to exclude columns
    projection = {"_id": 0}  # Always exclude MongoDB's `_id`
    if exclude_columns:
        for col in exclude_columns:
            projection[col] = 0  # Set to 0 to exclude the column

    for collection_name in filtered_collections:
        print(f"\n📂 Collection: {collection_name}")
        print("-" * 50)

        collection = db[collection_name]
        documents = list(collection.find({}, projection).limit(limit))

        if not documents:
            print("⚠️ No data found in this collection.")
            continue

        # Convert to DataFrame for better readability
        df = pd.DataFrame(documents)
        print(df.to_string(index=False))  # Pretty print DataFrame
        print("-" * 50)

### 📌 CHECK IF EXPERIMENT EXISTS ###
def experiment_exists(experiment: str) -> bool:
    """
    Checks if an experiment exists in MongoDB by verifying at least one collection.

    - Experiments typically have multiple collections (e.g., `{experiment}_errors`, `{experiment}_results`).
    - If at least one collection exists, we assume the experiment is set up.

    Returns:
    - `True` if at least one collection exists, otherwise `False`.
    """
    experiment_collections = [
        f"{experiment}_errors",
        f"{experiment}_results",
        f"{experiment}_best_params"
    ]

    # Check if any of the expected collections exist
    existing_collections = db.list_collection_names()
    
    return any(col in existing_collections for col in experiment_collections)


### 📌 SETUP EXPERIMENT COLLECTION ###
def setup_experiment_collection(experiment: str):
    """Sets up MongoDB collections for a new experiment inside the database."""
    
    if experiment_exists(experiment):
        raise FileExistsError(f"🚨 Experiment '{experiment}' already exists in MongoDB.")

    # Create collections inside the database
    db.create_collection(f"{experiment}_errors")
    db.create_collection(f"{experiment}_results")
    db.create_collection(f"{experiment}_best_params")

    # Indexing for performance
    db[f"{experiment}_errors"].create_index("task_id")
    db[f"{experiment}_results"].create_index("model_name")
    db[f"{experiment}_best_params"].create_index("model_name")

    print(f"✅ MongoDB Collections 'errors', 'results', 'best_params' was created for experiment '{experiment}' ")


### 📌 SAVE ERRORS TO MONGODB ###
def save_errors_to_db(experiment: str, model_name: str, test_results: dict[int, list[CodeEvaluationResult]]):
    """Saves errors to MongoDB in the '{experiment}_errors' collection."""
    collection = db[f"{experiment}_errors"]

    errors = []
    for task_id, results in test_results.items():
        for result in results:
            if not result.passed:  # Only log errors
                errors.append({
                    "model_name": model_name,
                    "task_id": task_id,
                    "candidate_id": result.candidate_id,
                    "metric": result.metric,
                    "code_candidate": result.code,
                    "test_result": result.test_result,
                    "error_type": result.error_type,
                    "error_msg": result.error_msg,
                    "stderr": result.compiler_msg.stderr if result.compiler_msg else "N/A",
                    "stdout": result.compiler_msg.stdout if result.compiler_msg else "N/A"
                })

    if errors:
        collection.insert_many(errors)
        print(f"⚠️ Errors logged in MongoDB for model '{model_name}' under experiment '{experiment}'.")  


### 📌 SAVE RESULTS TO MONGODB ###
def save_results_to_db(experiment: str, model_name: str, seed: int, temperature: float, top_p: float, top_k: int, ks: list[int], metric: str, pass_at_k: dict[int, float]):
    """Saves results to MongoDB in the '{experiment}_results' collection."""
    collection = db[f"{experiment}_results"]

    result_doc = {
        "model_name": model_name,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "ks": ks,
        "metric": metric,
        "pass_at_k": pass_at_k
    }

    collection.insert_one(result_doc)
    print(f"✅ Results saved to MongoDB for model '{model_name}' under experiment '{experiment}'.")  


### 📌 SAVE BEST PARAMS TO MONGODB ###
def save_best_params_to_db(experiment: str, model_name: str, optimizer_metric: str, best_params: Run):
    """Saves the best hyperparameters for a given model in MongoDB."""
    collection = db[f"{experiment}_best_params"]

    # Remove old best params if they exist
    collection.delete_many({"model_name": model_name, "optimizer_metric": optimizer_metric})

    # Insert new best params
    collection.insert_one({
        "model_name": model_name,
        "optimizer_metric": optimizer_metric,
        "temperature": best_params.temperature,
        "top_p": best_params.top_p,
        "top_k": best_params.top_k,
        "seed": best_params.seed,
    })

    print(f"✅ Best parameters saved in MongoDB for model '{model_name}' under experiment '{experiment}'.")  


### 📌 CHECK IF BEST PARAMS EXIST ###
def get_best_params(experiment: str, model):
    """Checks if best parameters from validation exist in MongoDB."""
    collection = db[f"{experiment}_best_params"]

    best_params = collection.find_one({"model_name": model["name"]})

    if best_params:
        print(f"✅ Using existing best parameters for model '{model['name']}': {best_params}")
        return Run(
            temperature=best_params["temperature"],
            top_p=best_params["top_p"],
            top_k=best_params["top_k"],
            seed=best_params["seed"],
        )
    
    print(f"⚠️ No best parameters found for model '{model['name']}' under experiment '{experiment}'.")
    return None


### 📌 LOAD COLLECTION FROM MONGODB INTO PANDAS ###
def load_collection_from_db(experiment: str, collection_type: str):
    """Loads a MongoDB collection into a Pandas DataFrame."""
    collection_name = f"{experiment}_{collection_type}"
    collection = db[collection_name]
    
    data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB ObjectId

    if not data:
        raise ValueError(f"Collection '{collection_name}' is empty in MongoDB.")

    return pd.DataFrame(data)


### 📌 VISUALIZE ERRORS AS A BAR CHART ###
def bar_chart_errors_by_type(experiment: str):
    """Plots a bar chart of error types from MongoDB."""
    df_errors = load_collection_from_db(experiment, "errors")

    error_counts = df_errors["error_msg"].value_counts()

    plt.figure(figsize=(8, 5))
    error_counts.plot(kind="bar", color="red", alpha=0.7)
    plt.xlabel("Error msg")
    plt.ylabel("Count")
    plt.title(f"Frequency of Error msg- {experiment}")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

### 📌 VISUALIZE RESULTS AS A BAR CHART ###
def visualize_results(experiment: str):
    """Loads and visualizes model performance from MongoDB."""
    
    # Load collection safely
    df_results = load_collection_from_db(experiment, "results")

    # **Check if the collection is empty**
    if df_results.empty:
        print(f"⚠️ No results found for experiment '{experiment}'. Skipping visualization.")
        return

    # Ensure pass_at_k is properly formatted
    df_results["pass_at_k"] = df_results["pass_at_k"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    # Extract pass@k values
    pass_at_k_list = []
    for _, row in df_results.iterrows():
        for k, value in row["pass_at_k"].items():
            pass_at_k_list.append({"Model": row["model_name"], "k": int(k), "Pass@k": value})

    df_pass_at_k = pd.DataFrame(pass_at_k_list)

    # **Check if extracted data is empty before plotting**
    if df_pass_at_k.empty:
        print(f"⚠️ No valid 'pass@k' data found for experiment '{experiment}'. Skipping visualization.")
        return

    # Pivot table for visualization
    pivot_df = df_pass_at_k.pivot(index="k", columns="Model", values="Pass@k")

    # Plot results
    pivot_df.plot(kind="bar", figsize=(10, 6), alpha=0.8)
    plt.xlabel("k (Number of Samples Considered)")
    plt.ylabel("Pass@k Score")
    plt.title(f"Model Performance (Pass@k) - {experiment}")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Models", loc="upper left")
    plt.show()


### 📌 EXPORT DATA FROM MONGODB ###
def export_collection(experiment: str, collection_type: str, file_format="json"):
    """
    Exports a MongoDB collection to JSON or CSV.

    Parameters:
        - experiment (str): The name of the experiment.
        - collection_type (str): The type of collection to export ('errors', 'results', 'best_params').
        - file_format (str): The output format ("json" or "csv"). Default is "json".
    """

    EXPORT_DIR = os.path.join(project_root, "notebooks/few-shot/exports")
    os.makedirs(EXPORT_DIR, exist_ok=True)

    collection_name = f"{experiment}_{collection_type}"
    collection = db[collection_name]
    
    data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB ObjectId

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


def delete_collection(experiment: str, collection_type: str):
    """
    Deletes a specific collection related to an experiment.

    Parameters:
    - experiment (str): The name of the experiment.
    - collection_type (str): The type of collection to delete (e.g., 'errors', 'results', 'best_params').

    Example:
        delete_collection("GPT4_signature_exp1", "results")
    """
    collection_name = f"{experiment}_{collection_type}"

    if collection_name not in db.list_collection_names():
        print(f"⚠️ Collection '{collection_name}' does not exist in the database.")
        return

    confirmation = input(f"❗ Are you sure you want to delete collection '{collection_name}'? (yes/no): ").strip().lower()
    
    if confirmation == "yes":
        db[collection_name].drop()
        print(f"✅ Collection '{collection_name}' has been deleted successfully.")
    else:
        print("❌ Deletion cancelled.")

from pymongo import MongoClient

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)

# Database Name
DB_NAME = "few_shot_experiments"
db = client[DB_NAME]

def delete_experiment(experiment: str):
    """
    Deletes an entire experiment by removing all its associated collections.

    Parameters:
    - experiment (str): The experiment name.

    Example:
        delete_experiment("GPT4_signature_exp1")
    """
    # List expected collections for the experiment
    experiment_collections = [
        f"{experiment}_errors",
        f"{experiment}_results",
        f"{experiment}_best_params"
    ]

    # Get existing collections
    existing_collections = db.list_collection_names()

    # Check if the experiment exists
    collections_to_delete = [col for col in experiment_collections if col in existing_collections]

    if not collections_to_delete:
        print(f"⚠️ No collections found for experiment '{experiment}'. Nothing to delete.")
        return

    # Confirm before deleting
    confirmation = input(f"❗ Are you sure you want to delete experiment '{experiment}'? (yes/no): ").strip().lower()
    
    if confirmation == "yes":
        for collection in collections_to_delete:
            db[collection].drop()
            print(f"✅ Deleted collection: {collection}")
        
        print(f"🗑️ Experiment '{experiment}' has been fully deleted.")
    else:
        print("❌ Deletion cancelled.")
