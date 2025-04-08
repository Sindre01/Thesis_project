import re
from matplotlib import pyplot as plt
import pandas as pd

def make_categories_pie_chart(df, title="Error Categories Pie Chart"):
    # Plot pie chart
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _ = ax.pie(
        df["count"],
        labels=None,
        startangle=90
    )

    # Add legend with percentages
    labels = [f"{row['category']} — {row['percentage']:.1f}%" for _, row in df.iterrows()]
    ax.legend(
        wedges, labels,
        title=title,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=10
    )

    # Final layout
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# Create a DataFrame of unique error categories
def get_error_category_counts(error_df, column_name):
    filtered = error_df

    all_categories = []

    for categories in filtered[column_name]:
        if isinstance(categories, (set, list)):
            all_categories.extend(categories)
        elif isinstance(categories, str):

            all_categories.extend([cat.strip() for cat in categories.split(",")])
        else:
            all_categories.append(str(categories)) 

    # Count
    category_series = pd.Series(all_categories)
    category_counts = category_series.value_counts().reset_index()
    category_counts.columns = ["category", "count"]
    total = category_counts["count"].sum()
    category_counts["percentage"] = category_counts["count"] / total * 100

    return category_counts

## Make categorization of syntax, semantic and test errors  
def categorize_syntax_error(stderr):
    match = re.search(r'Error:\s*(.*?)(?:\n|:)', stderr)

    if match:
        return match.group(1).strip()
    
    if "expected node" in stderr.lower():
        return "Unexpected node"
    
    if "code is not compile ready" in stderr.lower():
        return "Not compile ready"
    
    return match.group(1).strip() if match else "Other syntax error"


def categorize_semantic_errors(messages):
    categorized_errors = set()

    for msg in messages:
        msg_lower = msg.lower()

        if "unable to resolve type" in msg_lower or "failed to resolve symbol" in msg_lower:
            categorized_errors.add("Unresolved symbol")

        elif "arrow from" in msg_lower and "is not allowed" in msg_lower:
            categorized_errors.add("Invalid connection")

        elif "negative context production" in msg_lower:
            categorized_errors.add("Invalid context dependency")

        elif "function header" in msg_lower:
            categorized_errors.add("Invalid function header")

        elif "leaf node" in msg_lower:
            categorized_errors.add("Invalid AST structure")

        elif "expected function or event" in msg_lower:
            categorized_errors.add("Expected function or event")

        elif "compiler plugin encountered errors" in msg_lower:
            categorized_errors.add("Compiler plugin error")


    if not categorized_errors:
        return {"Other Semantic Error"}

    return categorized_errors

def categorize_test_errors(test_result):
    if test_result:
    
        total = 3
        passed = 0
        for test in test_result.get("test_results", []):
            for assertion in test.get("assertions", []):
                if assertion.get("kind") == "Passed":
                    passed += 1
        return f"{passed}/{total}"
    return "0/3"

# Extract different types of errors
def extract_semantic_errors(messages):
    semantic_errors = []
    for msg in messages:
        match = re.search(r"SemanticAnalysisError\(@\d+\): (.+?)(?:, backtrace|$)", msg)
        if match:
            semantic_errors.append(match.group(1).strip())
    return semantic_errors if semantic_errors else None

def extract_test_error(category, error_msg, test_result):
    message = f"{category} tests passed.\n "

    message += f"Test results:\n{extract_failed_tests(test_result, category)}\n"
    message += "\n".join(error_msg[:4])
    return message

def extract_failed_tests(test_result, category):
    if not isinstance(test_result, dict):
        return "No test results found."

    failed_msgs = []
    for test in test_result.get("test_results", []):
        for assertion in test.get("assertions", []):
            if assertion.get("kind") == "Failed":
                expected = assertion.get("expect")
                actual = assertion.get("actual")
                failed_msgs.append(f"Failed test: expected `{expected}`, got `{actual}`")
    if not failed_msgs and category == "3/3":
        return "All passed."
    
    return "\n".join(failed_msgs)