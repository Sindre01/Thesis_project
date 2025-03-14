import pandas as pd
from my_packages.db_service import db

def update_all_colelctions(update_dict: dict):
    """
    Updates all collections in the database with the provided update dictionary.
    """
    collections = db.list_collection_names()
    for collection_name in collections:
        collection = db[collection_name]

        for field, value in update_dict.items():
            # Only update documents where the field does NOT already exist
            result = collection.update_many(
                {field: {"$exists": False}},
                {"$set": {field: value}}
            )
            print(f"✅ Updated {result.modified_count} documents in collection '{collection_name}' with field '{field}' = '{value}'")

def confirm_rerun(
        experiment: str, 
        model_name: str, 
        eval_method: str,
        phase: str,
        db_connection=None
    ) -> bool:
    """
    Checks if a model for an eval_method already exists in the results collection of an experiment.
    If the model exists, asks the user whether to re-run the experiment.
    If 'yes', deletes the old results and testing errors, and runs the experiment again.

    Parameters:
    - experiment (str): The experiment name.
    - model_name (str): The model to check.
    - eval_method (str): The evaluation method (e.g., "hold_out", "3_fold").

    Returns:
    - True if the experiment should proceed, False otherwise.
    """
    if db_connection is None:
        db_connection = db
    if phase == "testing":
        collection_type = "results"
    elif phase == "validation":
        collection_type = "best_params"
    else:
        raise ValueError("Phase must be either 'testing' or 'validation'.")
    
    collection = db[f"{experiment}_{collection_type}"]
    errors_collection = db[f"{experiment}_errors"]
    # params_collection = db[f"{experiment}_best_params"]

    # Check if model exists in the {collection_type} collection
    existing_entry = collection.find_one({"model_name": model_name, "eval_method": eval_method})

    if existing_entry:
        print(f"⚠️ Model '{model_name}' already exists in {eval_method} experiment '{experiment}'.")
        user_input = input(f"❓ Do you want to re-run the experiment and delete {collection_type} and {phase} errors for this model? (yes/no):  ").strip().lower()

        if user_input == "yes":
            print(f"==== Cleaning experiment for model '{model_name}'======")
            
            # Delete existing entries in {collection_type} and errors
            deleted = collection.delete_many({"model_name": model_name, "eval_method": eval_method}).deleted_count
            errors_deleted = errors_collection.delete_many({"model_name": model_name, "eval_method": eval_method, "phase": phase}).deleted_count

            print(f"🗑️ Deleted {deleted} previous {collection_type} & {errors_deleted} {phase} errors for '{model_name}'.")
            print("Ready to re-run the experiment.\n")
            return True
        else:
            print(f"❌ Skipping {phase} for model '{model_name}'.\n")
            return False
    
    # If model does not exist, proceed with the experiment
    print(f"⚠️ No {collection_type} found for model '{model_name}' in {eval_method} experiment '{experiment}_{collection_type}'.\n")

    cleanup_errors = errors_collection.delete_many({"model_name": model_name, "eval_method": eval_method, "phase": phase}).deleted_count # Cleanup previous saved {phase} errors, due to interuptions
    if cleanup_errors:
        print(f"🗑️ Deleted {cleanup_errors} previous {phase} errors for '{model_name}'.\n")
    return True

### 📌 SETUP EXPERIMENT COLLECTION ###
def setup_experiment_collection(
        experiment: str,
        db_connection = None
    )-> None:
    """Sets up MongoDB collections for a new experiment inside the database."""
    if db_connection is None:
        db_connection = db
    if experiment_exists(experiment, db_connection):
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


    

def run_experiment_quality_checks(
        experiment: str, 
        prompt_user = True, 
        eval_method: str = "hold_out",
        db_connection=None
    ) -> bool:
    """
    Runs quality checks on an experiment's collections and prints errors if they occur.
    
    Checks:
      1. If no best parameters exist for a model, then no validation errors should exist.
      2. If no results exist for a model, then no testing errors should exist.
      3. If no best parameters exist for a model, then no results should exist.
    
    This function prints any errors that occur, or a success message for each model.

    Prompt user to fix issues by deleting errors or results if necessary.
    """
    if db_connection is None:
        db_connection = db
    # Define collections for this experiment.
    best_params_collection = db[f"{experiment}_best_params"]
    results_collection = db[f"{experiment}_results"]
    errors_collection = db[f"{experiment}_errors"]
    errors_found = False

    models = list_models_for_experiment(experiment, query_filter={"eval_method": eval_method})
    print(f"\n=== Running quality checks for {eval_method} experiment: {experiment} ===")
    if models:
        print("Models found in DB for experiment:", models)
    else:
        print(f"No models found for {eval_method} experiment '{experiment}'.")
    print_msgs = []
    for model in models:
        # Count documents in each collection for the given model.
        best_params_count = best_params_collection.count_documents({"model_name": model}) # currently only hold_out for best_params
        validation_errors_count = errors_collection.count_documents({"model_name": model, "phase": "validation"}) # currently only hold_out for validation errors
        
        results_count = results_collection.count_documents({"model_name": model, "eval_method": eval_method})
        testing_errors_count = errors_collection.count_documents({"model_name": model, "phase": "testing", "eval_method": eval_method})
        
        # Check 1: If no best parameters exist, then there should be no validation errors.
        if best_params_count == 0 and validation_errors_count > 0:
            user_input="no"
            if prompt_user:
                user_input = input(f"❌ Error for model '{model}': No best parameters exist, but found {validation_errors_count} validation error(s).\n\n❓ Do you want to delete validation errors for '{model} on '{experiment}'? (yes/no): ").strip().lower()
            
            if user_input == "yes":
                errors_deleted = errors_collection.delete_many({"model_name": model, "phase": "validation"}).deleted_count
                print_msgs.append(f"⚠️ Found Error for model '{model}': No best parameters exist, but found {validation_errors_count} validation error(s). \nHowever these where fixed by deleting {errors_deleted} validation errors for '{model}'.")
                
            else:
                print_msgs.append(f"❌ Error for model '{model}': No best parameters exist, but found {validation_errors_count} validation error(s).")
            errors_found = True
        
        # Check 2: If no results exist, then there should be no testing errors.
        if results_count == 0 and testing_errors_count > 0:
            user_input="no"
            if prompt_user:
                user_input = input(f"❌ Error for model '{model}': No best parameters exist, but found {testing_errors_count} testing error(s).\n\n❓ Do you want to delete testing errors for '{model} on '{experiment}'? (yes/no): ").strip().lower()

            if user_input == "yes":
                errors_deleted = errors_collection.delete_many({"model_name": model, "phase": "testing","eval_method": eval_method}).deleted_count
                print_msgs.append(f"⚠️ Found Error for model '{model}': No best parameters exist, but found {testing_errors_count} testing error(s). \nHowever these where fixed by deleting {errors_deleted} testing errors for '{model}'.")
                
            else:
                print_msgs.append(f"❌ Error for model '{model}': No results exist, but found {testing_errors_count} testing error(s).")
            errors_found = True
        
        # Check 3: If no best parameters exist, then there should be no results.
        if best_params_count == 0 and results_count > 0:
            user_input="no"
            if prompt_user:
                user_input = input(f"❌ Error for model '{model}': No best parameters exist, but found {results_count} result(s).\n\n❓ Do you want to delete result(s) for '{model} on '{experiment}'? (yes/no): ").strip().lower()

            if user_input == "yes":
                results_deleted = results_collection.delete_many({"model_name": model, "eval_method": eval_method}).deleted_count
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


def run_quality_checks_for_all_experiments(prompt_user = True) -> bool:
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
        errors_found = run_experiment_quality_checks(experiment, prompt_user=prompt_user)
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



def list_models_for_experiment(
        experiment: str,
        query_filter: dict | None = None
    ) -> list[str]:
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
            if query_filter:
                models_in_coll = db[coll_name].distinct("model_name", filter=query_filter)
            else:
                models_in_coll = db[coll_name].distinct("model_name")
            models.update(models_in_coll)
    return list(models)

def pretty_print_experiment_collections(
        experiment: str, 
        limit=5, 
        exclude_columns=["stderr", "stdout", "code_candidate"],
        db_connection=None,
        filter={}
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

        documents = list(collection.find(filter, projection).limit(limit))


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