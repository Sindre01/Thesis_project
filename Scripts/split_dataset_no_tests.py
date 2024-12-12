import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from Analysis.analyze_datasets import analyze_library_distribution, analyze_instance_distribution, analyze_visual_node_types_distribution

# Load the dataset
with open('../Data/MBPP_transformed_code_examples/sanitized-MBPP-midio.json', 'r') as file:
    data = json.load(file)

# Convert data to a NumPy array for indexing
data = np.array(data)

# Extract library_functions and textual_instance_types for each sample
combined_labels_list = []
library_functions_list = []
textual_instance_types_list = []
for item in data:
    # Process library_functions
    library_functions = item.get('library_functions', [])
    library_functions = [func.replace('root.std.', '') for func in library_functions]
    item['library_functions'] = library_functions  # Update the item for later use
    library_functions_list.append(library_functions)
    
    # Process textual_instance_types
    textual_instance_types = item.get('textual_instance_types', [])
    textual_instance_types_list.append(textual_instance_types)
    
    # Combine both for multilabel stratification
    combined_labels = library_functions + textual_instance_types
    combined_labels_list.append(combined_labels)

# Count the frequency of each label in the entire dataset
label_counts = Counter(label for labels in combined_labels_list for label in labels)

# Identify samples with labels used only once and ensure they are in the training set
unique_label_samples = set()
for idx, labels in enumerate(combined_labels_list):
    if any(label_counts[label] == 1 for label in labels):
        unique_label_samples.add(idx)

# Prepare labels for stratification
mlb = MultiLabelBinarizer()
labels_array = mlb.fit_transform(combined_labels_list)

# Initialize lists for indices
training_indices = list(unique_label_samples)
remaining_indices = list(set(range(len(data))) - unique_label_samples)

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
    remaining_labels = labels_array[remaining_indices]
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

# Now, split temp into validation and test sets
# Compute the proportion of validation and test sizes relative to temp data
temp_data_size = len(temp_idx)
if temp_data_size > 0:
    eval_ratio = desired_eval_size / (desired_eval_size + desired_test_size)
    eval_size = int(temp_data_size * eval_ratio)
    test_size = temp_data_size - eval_size

    msss_temp = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=eval_size,
        test_size=test_size,
        random_state=42
    )

    temp_data = data[temp_idx]
    temp_labels = labels_array[temp_idx]
    eval_idx_relative, test_idx_relative = next(msss_temp.split(temp_data, temp_labels))

    # Map back to original indices
    validation_indices = [temp_idx[i] for i in eval_idx_relative]
    test_indices = [temp_idx[i] for i in test_idx_relative]
else:
    validation_indices = []
    test_indices = []

# Ensure validation and test datasets do not contain any labels not in training dataset
training_labels = set(label for idx in training_indices for label in combined_labels_list[idx])

# Filter validation set
validation_indices_filtered = []
for idx in validation_indices:
    labels = combined_labels_list[idx]
    if set(labels).issubset(training_labels):
        validation_indices_filtered.append(idx)
    else:
        # Move sample to training set if it contains unseen labels
        training_indices.append(idx)

# Update validation_indices
validation_indices = validation_indices_filtered

# Filter test set
test_indices_filtered = []
for idx in test_indices:
    labels = combined_labels_list[idx]
    if set(labels).issubset(training_labels):
        test_indices_filtered.append(idx)
    else:
        # Move sample to training set if it contains unseen labels
        training_indices.append(idx)

# Update test_indices
test_indices = test_indices_filtered

# Collect any samples that were moved from validation or test sets to training set
remaining_temp_indices = set(temp_idx) - set(validation_indices) - set(test_indices)
training_indices.extend(remaining_temp_indices)

# Split the data
train_data = data[training_indices]
validation_data = data[validation_indices]
test_data = data[test_indices]

# Save the splits into files
with open('train_dataset.json', 'w') as train_file:
    json.dump(train_data.tolist(), train_file, indent=4)

with open('validation_dataset.json', 'w') as eval_file:
    json.dump(validation_data.tolist(), eval_file, indent=4)

with open('test_dataset.json', 'w') as test_file:
    json.dump(test_data.tolist(), test_file, indent=4)

# Print the number of samples in each dataset
print(f"Number of samples in training dataset: {len(train_data)}")
print(f"Number of samples in validation dataset: {len(validation_data)}")
print(f"Number of samples in test dataset: {len(test_data)}")
print(f"Number of samples used accross split: {len(test_data) + len(train_data) + len(validation_data)}")

# Identify labels only in training dataset
labels_in_eval = set(label for idx in validation_indices for label in combined_labels_list[idx])
labels_in_test = set(label for idx in test_indices for label in combined_labels_list[idx])
labels_only_in_training = training_labels - labels_in_eval - labels_in_test

print("\nLabels only in training dataset:")
for label in sorted(labels_only_in_training):
    print(label)

# Identify labels only in training dataset
labels_in_eval = set(label for idx in validation_indices for label in combined_labels_list[idx])
labels_in_test = set(label for idx in test_indices for label in combined_labels_list[idx])
labels_only_in_training = training_labels - labels_in_eval - labels_in_test

print("\nLabels only in training dataset:")
for label in sorted(labels_only_in_training):
    print(label)

analyze_library_distribution(train_data, validation_data, test_data)
analyze_instance_distribution(train_data, validation_data, test_data)
analyze_visual_node_types_distribution(train_data, validation_data, test_data)