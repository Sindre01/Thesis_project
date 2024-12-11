import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Load the dataset
with open('MBPP_transformed_code_examples/sanitized-MBPP-midio.json', 'r') as file:
    data = json.load(file)

# Convert data to a NumPy array for indexing
data = np.array(data)

# Extract library functions for each sample
library_functions_list = []
for item in data:
    library_functions = item.get('library_functions', [])
    # Remove 'root.std.' prefix if present
    library_functions = [func.replace('root.std.', '') for func in library_functions]
    item['library_functions'] = library_functions  # Update the item for later use
    library_functions_list.append(library_functions)

# Count the frequency of each library function in the entire dataset
function_counts = Counter(func for funcs in library_functions_list for func in funcs)

# Identify samples with library functions used only once and ensure they are in the training set
unique_function_samples = set()
for idx, funcs in enumerate(library_functions_list):
    if any(function_counts[func] == 1 for func in funcs):
        unique_function_samples.add(idx)

# Prepare labels for stratification
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(library_functions_list)

# Initialize lists for indices
training_indices = list(unique_function_samples)
remaining_indices = list(set(range(len(data))) - unique_function_samples)

# Determine the desired sizes
total_samples = len(data)
desired_training_size = int(total_samples * 0.7)
desired_eval_size = int(total_samples * 0.2)
desired_test_size = total_samples - desired_training_size - desired_eval_size  # To account for rounding

# Calculate how many more samples we need in training
additional_training_needed = desired_training_size - len(training_indices)

# Perform stratified split on the remaining data to get training and temp (eval + test)
if additional_training_needed > 0 and remaining_indices:
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=additional_training_needed,
        test_size=None,
        random_state=42
    )
    # Indices need to be adjusted to match remaining_indices
    remaining_data = data[remaining_indices]
    remaining_labels = labels[remaining_indices]
    train_idx, temp_idx = next(msss.split(remaining_data, remaining_labels))

    # Map back to original indices
    train_idx = [remaining_indices[i] for i in train_idx]
    temp_idx = [remaining_indices[i] for i in temp_idx]

    # Add to training indices
    training_indices.extend(train_idx)
else:
    # All data is in training set
    training_indices = list(range(len(data)))
    temp_idx = []

# Now, split temp into evaluation and test sets
# Compute the proportion of evaluation and test sizes relative to temp data
temp_data_size = len(temp_idx)
if temp_data_size > 0:
    eval_size = desired_eval_size
    test_size = desired_test_size

    # Adjust if there are not enough samples
    if eval_size + test_size > temp_data_size:
        eval_size = int(temp_data_size * (desired_eval_size / (desired_eval_size + desired_test_size)))
        test_size = temp_data_size - eval_size

    msss_temp = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=eval_size,
        test_size=test_size,
        random_state=42
    )

    temp_data = data[temp_idx]
    temp_labels = labels[temp_idx]
    eval_idx_relative, test_idx_relative = next(msss_temp.split(temp_data, temp_labels))

    # Map back to original indices
    evaluation_indices = [temp_idx[i] for i in eval_idx_relative]
    test_indices = [temp_idx[i] for i in test_idx_relative]
else:
    evaluation_indices = []
    test_indices = []

# Ensure evaluation and test datasets do not contain any library functions not in training dataset
training_functions = set(func for idx in training_indices for func in library_functions_list[idx])

# Filter evaluation set
evaluation_indices_filtered = []
for idx in evaluation_indices:
    funcs = library_functions_list[idx]
    if set(funcs).issubset(training_functions):
        evaluation_indices_filtered.append(idx)
    else:
        # Move sample to training set if it contains unseen functions
        training_indices.append(idx)

# Update evaluation_indices
evaluation_indices = evaluation_indices_filtered

# Filter test set
test_indices_filtered = []
for idx in test_indices:
    funcs = library_functions_list[idx]
    if set(funcs).issubset(training_functions):
        test_indices_filtered.append(idx)
    else:
        # Move sample to training set if it contains unseen functions
        training_indices.append(idx)

# Update test_indices
test_indices = test_indices_filtered

# Collect any samples that were moved from evaluation or test sets to training set
# Remaining samples in temp that couldn't be added to evaluation or test sets
remaining_temp_indices = set(temp_idx) - set(evaluation_indices) - set(test_indices)
training_indices.extend(remaining_temp_indices)

# Split the data
train_data = data[training_indices]
evaluation_data = data[evaluation_indices]
test_data = data[test_indices]

# Save the splits into files
with open('train_dataset.json', 'w') as train_file:
    json.dump(train_data.tolist(), train_file, indent=4)

with open('evaluation_dataset.json', 'w') as eval_file:
    json.dump(evaluation_data.tolist(), eval_file, indent=4)

with open('test_dataset.json', 'w') as test_file:
    json.dump(test_data.tolist(), test_file, indent=4)

# Verify the distribution
def get_label_distribution(data_subset):
    subset_labels = []
    for item in data_subset:
        library_functions = item.get('library_functions', [])
        subset_labels.extend(library_functions)
    return subset_labels

train_distribution = Counter(get_label_distribution(train_data))
eval_distribution = Counter(get_label_distribution(evaluation_data))
test_distribution = Counter(get_label_distribution(test_data))

# Generate bar charts comparing the three datasets with total counts in the background
all_functions = sorted(function_counts.keys())
x = np.arange(len(all_functions))  # the label locations
width = 0.2  # the width of the bars

total_counts = [function_counts.get(func, 0) for func in all_functions]
train_counts = [train_distribution.get(func, 0) for func in all_functions]
eval_counts = [eval_distribution.get(func, 0) for func in all_functions]
test_counts = [test_distribution.get(func, 0) for func in all_functions]

plt.figure(figsize=(16, 8))
fig, ax = plt.subplots(figsize=(16, 8))

# Plot the total counts as background bars
rects_total = ax.bar(x, total_counts, width*1.5, color='lightgrey', label='Total Count')

# Plot the training, evaluation, and test counts on top
rects1 = ax.bar(x - width, train_counts, width, label='Training Set')  # , color='skyblue')
rects2 = ax.bar(x, eval_counts, width, label='Evaluation Set')  # , color='sandybrown')
rects3 = ax.bar(x + width, test_counts, width, label='Test Set')  # , color='lightgreen')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Counts')
ax.set_title('Library Function Distribution in Training, Evaluation, and Test Sets')
ax.set_xticks(x)
ax.set_xticklabels(all_functions, rotation=90)
ax.legend()

plt.tight_layout()
plt.show()

# Identify library functions only in training dataset
functions_in_eval = set(eval_distribution.keys())
functions_in_test = set(test_distribution.keys())
functions_only_in_training = set(train_distribution.keys()) - functions_in_eval - functions_in_test
print("\nLibrary functions only in training dataset:")
for func in functions_only_in_training:
    print(func)