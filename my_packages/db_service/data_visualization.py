from matplotlib import pyplot as plt
import plotly.graph_objects as go
from my_packages.db_service.data_export import load_collection_from_db

def bar_chart_errors_for_metric_filtered(experiment: str, model_name: str, metric: str):
    """
    Loads error data from the 'errors' collection for a given experiment,
    filters by model_name and metric, and plots a bar chart showing the total number
    of directly detected errors for each error type (syntax, semantic, tests).

    Parameters:
      - experiment (str): The experiment name.
      - model_name (str): The model name to filter errors on.
      - metric (str): The evaluation metric (e.g., "syntax", "semantic", "tests").
    """
    # Load error data from the experiment's errors collection
    df_errors = load_collection_from_db(experiment, "errors")
    if df_errors.empty:
        print(f"⚠️ No errors found for experiment '{experiment}'.")
        return

    # Filter errors by model and metric
    df_filtered = df_errors[(df_errors["model_name"] == model_name) & (df_errors["metric"] == metric)]
    
    if df_filtered.empty:
        print(f"⚠️ No errors found for model '{model_name}' and metric '{metric}' in experiment '{experiment}'.")
        return

    # Count errors per type
    error_counts = df_filtered["error_type"].value_counts().sort_index()

    # Create a bar chart of the error counts
    plt.figure(figsize=(8, 5))
    ax = error_counts.plot(kind="bar", color=["red", "orange", "blue"], alpha=0.8)
    plt.xlabel("Error Type", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(f"Error Distribution for {model_name} ({metric}) - {experiment}", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Annotate bars with their count values
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, color='black')
    
    plt.tight_layout()
    plt.show()

### 📌 VISUALIZE ERRORS AS A BAR CHART ###
def bar_chart_errors_by_type(experiment: str):
    """Plots a bar chart of error types from MongoDB."""
    df_errors = load_collection_from_db(experiment, "errors")

    error_counts = df_errors["error_type"].value_counts()

    plt.figure(figsize=(8, 5))
    error_counts.plot(kind="bar", color="red", alpha=0.7)
    plt.xlabel("Error type")
    plt.ylabel("Count")
    plt.title(f"Frequency of Error type- {experiment}")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

def visualize_error_flow_for_model(experiment: str, model_name: str):
    """
    Visualizes error flow for a specific model in an experiment using a Sankey diagram.
    
    Parameters:
      - experiment (str): The experiment name.
      - model_name (str): The model name to filter errors on.
      - total_candidates (int): Total number of candidate evaluations for this model.
      
    This function computes:
      - syntax_errors: Count of documents in the 'errors' collection with error_type "syntax"
      - semantic_errors: Count with error_type "semantic"
      - tests_errors: Count with error_type "tests"
      
    Then it calculates:
      - syntax_valid = total_candidates - syntax_errors
      - semantic_valid = syntax_valid - semantic_errors
      - tests_passed = semantic_valid - tests_errors
      
    And creates a Sankey diagram that shows the hierarchical flow.
    """
    # Load error data from the experiment's errors collection (assumes load_collection_from_db is defined)
    df_errors = load_collection_from_db(experiment, "errors")
    if df_errors.empty:
        print(f"⚠️ No errors found for experiment '{experiment}'.")
        return
    total_candidates = df_errors[df_errors["model_name"] == model_name].shape[0]
    # Filter errors for the specific model
    df_model = df_errors[df_errors["model_name"] == model_name]
    
    # Count error types (default to 0 if not present)
    syntax_errors = int(df_model[df_model["error_type"] == "syntax"].shape[0])
    semantic_errors = int(df_model[df_model["error_type"] == "semantic"].shape[0])
    tests_errors = int(df_model[df_model["error_type"] == "tests"].shape[0])
    
    # Calculate the number of candidates that passed each phase
    syntax_valid = total_candidates - syntax_errors
    semantic_valid = syntax_valid - semantic_errors
    tests_passed = semantic_valid - tests_errors

    # Define node labels
    labels = [
        "Total Candidates",    # Node 0
        "Syntax Valid",        # Node 1
        "Syntax Error",        # Node 2
        "Semantic Valid",      # Node 3
        "Semantic Error",      # Node 4
        "Tests Passed",        # Node 5
        "Tests Error"          # Node 6
    ]
    
    # Define the flow links (source, target, value)
    source = [
        0, 0,        # Total Candidates splits to Syntax Valid and Syntax Error
        1, 1,        # Syntax Valid splits to Semantic Valid and Semantic Error
        3, 3         # Semantic Valid splits to Tests Passed and Tests Error
    ]
    target = [
        1, 2,        # Total -> Syntax Valid, Total -> Syntax Error
        3, 4,        # Syntax Valid -> Semantic Valid, Syntax Valid -> Semantic Error
        5, 6         # Semantic Valid -> Tests Passed, Semantic Valid -> Tests Error
    ]
    values = [
        syntax_valid, syntax_errors,
        semantic_valid, semantic_errors,
        tests_passed, tests_errors
    ]
    
    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=["#a6cee3", "#1f78b4", "#e31a1c", "#33a02c", "#ff7f00", "#6a3d9a", "#b15928"]
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color=["green", "red", "green", "orange", "green", "red"]
        )
    )])
    
    fig.update_layout(title_text=f"Error Flow for {model_name} in Experiment {experiment}\nTotal Candidates: {total_candidates}", font_size=12)
    fig.show()