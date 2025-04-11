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
    
def make_categories_bar_chart(df, title="Error Categories Bar Chart"):
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create the bar chart using the 'category' column for x-axis and 'count' for the bar heights.
    bars = ax.bar(df["category"], df["count"], color='skyblue')

    # Annotate each bar with the corresponding percentage value.
    for bar, percentage in zip(bars, df["percentage"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height,
            f'{percentage:.1f}%',
            ha='center', va='bottom',
            fontsize=10
        )

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel("Error Category")
    ax.set_ylabel("Count")

    # Optional: Rotate x-axis labels for better readability if necessary.
    plt.xticks(rotation=45, ha="right")
    
    # Adjust layout and display the figure
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

        if "unable to resolve type" in msg_lower or "failed to reify declaration path" in msg_lower:
            categorized_errors.add("Type Resolution Error")

        if "failed to resolve symbol" in msg_lower:
            categorized_errors.add("Symbol Resolution Error")

        elif "arrow from" in msg_lower and "is not allowed" in msg_lower:
            categorized_errors.add("Invalid connection")

        # elif "negative context production" in msg_lower:
        #     categorized_errors.add("Invalid context dependency")

        elif "function header" in msg_lower:
            categorized_errors.add("Invalid function header")

        elif "leaf node" in msg_lower:
            categorized_errors.add("Invalid AST structure")

        elif "expected function or event" in msg_lower:
            categorized_errors.add("Expected function or event")

        elif "compiler plugin encountered errors" in msg_lower:
            categorized_errors.add("Compiler plugin error")

        elif "already exists in the symbol table" in msg_lower:
            categorized_errors.add("Duplicate symbol")


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
import re

def extract_semantic_errors(messages):
    semantic_errors = []
    pattern1 = r"SemanticAnalysisError\(@\d+\): (.+?)(?:, backtrace|$)"
    pattern2 = r"Error\s+.*?:\s*(.+?)(?:, backtrace|$)"
    
    for msg in messages:
        # Find all matches for pattern1
        matches1 = re.findall(pattern1, msg)
        # If no matches, try pattern2
        if not matches1:
            matches1 = re.findall(pattern2, msg)
        # Append all found matches (if any), stripped of whitespace
        semantic_errors.extend(match.strip() for match in matches1)
    
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