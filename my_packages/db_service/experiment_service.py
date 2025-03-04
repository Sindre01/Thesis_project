import pandas as pd
from my_packages.db_service import db

def confirm_testing_rerun(experiment: str, model_name: str) -> bool:
    """
    Checks if a model already exists in the results collection of an experiment.
    If the model exists, asks the user whether to re-run the experiment.
    If 'yes', deletes the old results and testing errors, and runs the experiment again.

    Parameters:
    - experiment (str): The experiment name.
    - model_name (str): The model to check.

    Returns:
    - True if the experiment should proceed, False otherwise.
    """

    results_collection = db[f"{experiment}_results"]
    errors_collection = db[f"{experiment}_errors"]
    # params_collection = db[f"{experiment}_best_params"]

    # Check if model exists in the results collection
    existing_entry = results_collection.find_one({"model_name": model_name})

    if existing_entry:
        print(f"⚠️ Model '{model_name}' already exists in experiment '{experiment}'.")
        user_input = input("❓ Do you want to re-run the experiment and delete results and testing errors for this model? (yes/no):  ").strip().lower()

        if user_input == "yes":
            print(f"==== Cleaning experiment for model '{model_name}'======")
            
            # Delete existing entries in results and errors
            results_deleted = results_collection.delete_many({"model_name": model_name}).deleted_count
            errors_deleted = errors_collection.delete_many({"model_name": model_name, "phase": "testing"}).deleted_count

            print(f"🗑️ Deleted {results_deleted} previous results & {errors_deleted} testing errors for '{model_name}'.")
            print("Ready to re-run the experiment.\n")
            return True
        else:
            print(f"❌ Skipping testing for model '{model_name}'.\n")
            return False
    
    # If model does not exist, proceed with the experiment
    print(f"⚠️ No results found for model '{model_name}' in experiment '{experiment}_best_params'.\n")

    cleanup_errors = errors_collection.delete_many({"model_name": model_name, "phase": "testing"}).deleted_count # Cleanup previous saved testing errors, due to interuptions
    if cleanup_errors:
        print(f"🗑️ Deleted {cleanup_errors} previous testing errors for '{model_name}'.\n")
    return True

def confirm_validation_rerun(experiment: str, model_name: str) -> bool:
    """
    Checks if a model already exists in the best params collection of an experiment.
    If the model exists, asks the user whether to re-run the experiment.
    If 'yes', deletes the old best params and validation errors, and runs the experiment again.

    Parameters:
    - experiment (str): The experiment name.
    - model_name (str): The model to check.

    Returns:
    - True if the experiment should proceed, False otherwise.
    """

    params_collection = db[f"{experiment}_best_params"]
    errors_collection = db[f"{experiment}_errors"]

    # Check if model exists in the results collection
    existing_entry = params_collection.find_one({"model_name": model_name})

    if existing_entry:
        print(f"⚠️ Model '{model_name}' already exists in experiment '{experiment}'.")
        user_input = input("❓ Do you want to re-run the experiment and delete best params and validation errors for this model? (yes/no): ").strip().lower()

        if user_input == "yes":
            print(f"==== Cleaning experiment for model '{model_name}'======")
            
            # Delete existing entries in results and errors
            params_deleted = params_collection.delete_many({"model_name": model_name}).deleted_count
            errors_deleted = errors_collection.delete_many({"model_name": model_name, "phase": "validation"}).deleted_count

            print(f"🗑️ Deleted {params_deleted} previous best params & {errors_deleted} validation errors for '{model_name}'.")
            print("Ready to re-run the experiment.\n")
            return True
        else:
            print(f"❌ Skipping validation for model '{model_name}'.\n")
            return False
    
    # If model does not exist, proceed with the experiment
    print(f"⚠️ No best params found for model '{model_name}' in experiment '{experiment}_best_params'. Ready to rerun! \n")

    cleanup_errors = errors_collection.delete_many({"model_name": model_name, "phase": "validation"}).deleted_count  # Cleanup previous saved validation errors, due to interuptions
    if cleanup_errors:
        print(f"🗑️ Deleted {cleanup_errors} previous validation errors for '{model_name}'. \n")
    return True


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


def experiment_exists(experiment: str, db_connection = None) -> bool:
    """
    Checks if an experiment exists in MongoDB by verifying at least one collection.

    - Experiments typically have multiple collections (e.g., `{experiment}_errors`, `{experiment}_results`).
    - If at least one collection exists, we assume the experiment is set up.

    Returns:
    - `True` if at least one collection exists, otherwise `False`.
    """
    if db_connection is None:
        db_connection = db

    experiment_collections = [
        f"{experiment}_errors",
        f"{experiment}_results",
        f"{experiment}_best_params"
    ]

    # Check if any of the expected collections exist
    existing_collections = db_connection.list_collection_names()
    
    return any(col in existing_collections for col in experiment_collections)


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




def remove_field(collection_name: str, field_name: str):
    """
    Removes the 'field_name' field from the results collection of an experiment.
    """
    results_collection = db[collection_name]
    if field_name not in results_collection.find_one({}):
        print(f"⚠️ Field '{field_name}' does not exist in the collection.")
        return
    result = results_collection.update_many({}, {"$unset": {field_name: ""}})
    print(f"Removed '{field_name}' field from {result.modified_count} documents.")

def rename_field(collection_name: str, old_field: str, new_field: str):
    """
    Renames the 'old_field' to 'new_field' in the results collection of an experiment.
    """
    results_collection = db[collection_name]
    if old_field not in results_collection.find_one({}):
        print(f"⚠️ Field '{old_field}' does not exist in the collection.")
        return
    result = results_collection.update_many({}, {"$rename": {old_field: new_field}})
    print(f"Renamed '{old_field}' to '{new_field}' in {result.modified_count} documents.")


    

def run_experiment_quality_checks(experiment: str) -> bool:
    """
    Runs quality checks on an experiment's collections and prints errors if they occur.
    
    Checks:
      1. If no best parameters exist for a model, then no validation errors should exist.
      2. If no results exist for a model, then no testing errors should exist.
      3. If no best parameters exist for a model, then no results should exist.
    
    This function prints any errors that occur, or a success message for each model.

    Prompt user to fix issues by deleting errors or results if necessary.
    """
    # Define collections for this experiment.
    best_params_collection = db[f"{experiment}_best_params"]
    results_collection = db[f"{experiment}_results"]
    errors_collection = db[f"{experiment}_errors"]
    errors_found = False

    models = list_models_for_experiment(experiment)
    print(f"\n=== Running quality checks for experiment: {experiment} ===")
    if models:
        print("Models found in DB for experiment::", models)
    else:
        print(f"No models found for experiment '{experiment}'.")
    print_msgs = []
    for model in models:
        # Count documents in each collection for the given model.
        best_params_count = best_params_collection.count_documents({"model_name": model})
        results_count = results_collection.count_documents({"model_name": model})
        validation_errors_count = errors_collection.count_documents({"model_name": model, "phase": "validation"})
        testing_errors_count = errors_collection.count_documents({"model_name": model, "phase": "testing"})
        
        
        # Check 1: If no best parameters exist, then there should be no validation errors.
        if best_params_count == 0 and validation_errors_count > 0:
            user_input = input(f"❌ Error for model '{model}': No best parameters exist, but found {validation_errors_count} validation error(s).\n\n❓ Do you want to delete validation errors for '{model} on '{experiment}'? (yes/no): ").strip().lower()

            if user_input == "yes":
                errors_deleted = errors_collection.delete_many({"model_name": model, "phase": "validation"}).deleted_count
                print_msgs.append(f"⚠️ Found Error for model '{model}': No best parameters exist, but found {validation_errors_count} validation error(s). \nHowever these where fixed by deleting {errors_deleted} validation errors for '{model}'.")
                
            else:
                print_msgs.append(f"❌ Error for model '{model}': No best parameters exist, but found {validation_errors_count} validation error(s).")
            errors_found = True
        
        # Check 2: If no results exist, then there should be no testing errors.
        if results_count == 0 and testing_errors_count > 0:
            user_input = input(f"❌ Error for model '{model}': No best parameters exist, but found {testing_errors_count} testing error(s).\n\n❓ Do you want to delete testing errors for '{model} on '{experiment}'? (yes/no): ").strip().lower()

            if user_input == "yes":
                errors_deleted = errors_collection.delete_many({"model_name": model, "phase": "testing"}).deleted_count
                print_msgs.append(f"⚠️ Found Error for model '{model}': No best parameters exist, but found {testing_errors_count} testing error(s). \nHowever these where fixed by deleting {errors_deleted} testing errors for '{model}'.")
                
            else:
                print_msgs.append(f"❌ Error for model '{model}': No results exist, but found {testing_errors_count} testing error(s).")
            errors_found = True
        
        # Check 3: If no best parameters exist, then there should be no results.
        if best_params_count == 0 and results_count > 0:
            user_input = input(f"❌ Error for model '{model}': No best parameters exist, but found {results_count} result(s).\n\n❓ Do you want to delete result(s) for '{model} on '{experiment}'? (yes/no): ").strip().lower()

            if user_input == "yes":
                results_deleted = results_collection.delete_many({"model_name": model}).deleted_count
                print_msgs.append(f"⚠️ Found Error for model '{model}': No best parameters exist, but found {results_count} result(s).\nHowever these where fixed by deleting {results_deleted} result(s) for '{model}'.")
                
            else:
                print_msgs.append(f"❌ Error for model '{model}': No best parameters exist, but found {results_count} result(s).")
            errors_found = True
        
        if not errors_found:
            print_msgs.append(f"✅ Models '{model}' passed all quality checks successfully.")
        
    if errors_found:
        print("\n".join(print_msgs))
    else:
        print("✅ All models passed quality checks successfully.")
    print("=== End of quality checks ===\n")
    return not errors_found


def run_quality_checks_for_all_experiments() -> bool:
    """
    Lists all experiments and models dynamically from the DB and runs quality checks.
    """
    experiments = list_experiments()
    if not experiments:
        print("No experiments found in the database.")
        return

    print("Found experiments:", experiments)
    errors_found = False
    
    for experiment in experiments:
        errors_found = run_experiment_quality_checks(experiment)
    return errors_found


def list_experiments() -> list[str]:
    """
    Dynamically list experiments based on collection names.
    Assumes collections follow the naming convention:
      {experiment}_best_params, {experiment}_results, {experiment}_errors
    """
    suffixes = ("_best_params", "_results", "_errors")
    experiments = set()
    for coll_name in db.list_collection_names():
        for suffix in suffixes:
            if coll_name.endswith(suffix):
                # Remove the suffix to get the experiment name.
                experiments.add(coll_name[:-len(suffix)])
                break
    return list(experiments)
def list_collections() -> list[str]:
    """
    Lists all collections in the MongoDB database.
    """
    return db.list_collection_names()



def list_models_for_experiment(experiment: str) -> list[str]:
    """
    Dynamically list model names for a given experiment.
    It queries all three collections for the experiment and takes the union.
    """
    models = set()
    collections_to_check = [f"{experiment}_best_params", 
                            f"{experiment}_results", 
                            f"{experiment}_errors"]
    existing_collections = set(db.list_collection_names())
    for coll_name in collections_to_check:
        if coll_name in existing_collections:
            # Get distinct model names from the collection.
            models_in_coll = db[coll_name].distinct("model_name")
            models.update(models_in_coll)
    return list(models)


def pretty_print_experiment_collections(
        experiment: str, 
        limit=5, 
        exclude_columns=["stderr", "stdout", "code_candidate"],
        db_connection=None
    ):
    """
    Prints all collections related to an experiment in a readable format.

    Parameters:
    - experiment (str): The experiment name.
    - limit (int): Number of documents to show per collection (default: 5).
    - exclude_columns (list): List of column names to exclude from output.
    """
    if db_connection is None:
        db_connection = db

    print(f"\n🔍 Existing collections for {experiment}: ")
    print("=" * 50)

    # List all collections for the experiment
    collections = db_connection.list_collection_names()
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
        print(f"\n📂 Collection: {collection_name} ")
        print("-" * 50)

        collection = db[collection_name]
        documents = list(collection.find({}, projection).limit(limit))

        if not documents:
            print("⚠️ No data found in this collection.")
            continue
        
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
        if collection_name == f"{experiment}_errors":
            extra_info = f"validation: {collection.count_documents({'phase': 'validation'})}, testing: {collection.count_documents({'phase': 'testing'})}"


        print(f"Total documents/rows: {collection.count_documents({})}      {extra_info}")
        print("-" * 50)

def extract_size(model: str):
    if ":" not in model:
        return "N/A"
    return model.split(":")[1].strip().split("-")[0]