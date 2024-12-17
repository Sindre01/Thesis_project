import json
import os
import numpy as np
import random
from collections import Counter
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def split(dataset, write_to_file = False, desired_training_size = 0.7, desired_eval_size = 0.2):
    # Convert data to a NumPy array for indexing
    data = np.array(dataset)

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

    # Determine remaining test_size
    total_samples = len(data)
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

    if write_to_file:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, '../../data')

        with open(f'{full_path}/train_dataset.json', 'w') as train_file:
            json.dump(train_data.tolist(), train_file, indent=4)
        
        with open(f'{full_path}/validation_dataset.json', 'w') as eval_file:
            json.dump(validation_data.tolist(), eval_file, indent=4)
        
        with open(f'{full_path}/test_dataset.json', 'w') as test_file:
            json.dump(test_data.tolist(), test_file, indent=4)

    # Identify labels only in training dataset
    labels_in_eval = set(label for idx in validation_indices for label in combined_labels_list[idx])
    labels_in_test = set(label for idx in test_indices for label in combined_labels_list[idx])
    labels_only_in_training = training_labels - labels_in_eval - labels_in_test

    # print("\nLabels only in training dataset:")
    # for label in sorted(labels_only_in_training):
    #     print(label)

#Splits dataset for few-shot prompting, where num-shots is number of samples to be included in train-set
import json
import numpy as np
import random
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import os

def split_on_shots(num_shots, eval_size_percentage, dataset, seed=62, write_to_file=False):
    # Set seeds for reproducibility
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    
    data = np.array(dataset)
    
    # Extract combined labels for each sample
    combined_labels_list = []
    for item in data:
        library_functions = item.get('library_functions', [])
        library_functions = [func.replace('root.std.', '') for func in library_functions]
        item['library_functions'] = library_functions
    
        textual_instance_types = item.get('textual_instance_types', [])
        item['textual_instance_types'] = textual_instance_types
    
        combined_labels = library_functions + textual_instance_types
        combined_labels_list.append(combined_labels)
    
    mlb = MultiLabelBinarizer()
    labels_array = mlb.fit_transform(combined_labels_list)
    all_indices = np.arange(len(data))
    
    # Build the training set with num_shots samples
    training_indices = []
    
    # Select training samples that cover as many labels as possible
    label_coverage = set()
    sample_labels = []
    for idx, labels in enumerate(combined_labels_list):
        sample_labels.append({'index': idx, 'labels': set(labels), 'num_labels': len(set(labels))})
    
    for sample in sorted(sample_labels, key=lambda x: x['num_labels'], reverse=True):
        if len(training_indices) >= num_shots:
            break
        if not sample['labels'].issubset(label_coverage):
            training_indices.append(sample['index'])
            label_coverage.update(sample['labels'])
    
    # If not enough, fill randomly
    if len(training_indices) < num_shots:
        remaining_indices = list(set(all_indices) - set(training_indices))
        random.shuffle(remaining_indices)
        training_indices.extend(remaining_indices[:num_shots - len(training_indices)])
    
    # Collect labels used in the training set
    training_labels = set(label for idx in training_indices for label in combined_labels_list[idx])
    remaining_indices = list(set(all_indices) - set(training_indices))
    
    # Filter samples that only have labels from the training set
    filtered_indices = [idx for idx in remaining_indices
                        if set(combined_labels_list[idx]).issubset(training_labels)]
    
    # Total number of possible samples after filtering
    total_possible_samples = len(filtered_indices)
    
    # Calculate the number of evaluation samples based on eval_size_percentage
    num_eval_samples = int(eval_size_percentage * total_possible_samples)
    num_eval_samples = max(num_eval_samples, 1)  # Ensure at least one sample
    
    # Remaining samples go to the test set
    num_test_samples = total_possible_samples - num_eval_samples
    
    # Shuffle filtered indices to randomize before splitting
    random.shuffle(filtered_indices)
    
    # Split the filtered data into evaluation and test sets
    if num_eval_samples > 0 and num_test_samples > 0:
        # Prepare data and labels for stratified splitting
        filtered_data = data[filtered_indices]
        filtered_labels = labels_array[filtered_indices]
        
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            train_size=num_eval_samples,
            test_size=num_test_samples,
            random_state=SEED
        )
        eval_indices_rel, test_indices_rel = next(msss.split(filtered_data, filtered_labels))
        eval_indices = [filtered_indices[i] for i in eval_indices_rel]
        test_indices = [filtered_indices[i] for i in test_indices_rel]
    else:
        # If not enough samples, assign all to evaluation or test set accordingly
        eval_indices = filtered_indices[:num_eval_samples]
        test_indices = filtered_indices[num_eval_samples:]
    
    # Ensure uniqueness and no overlap between sets
    assert len(set(training_indices) & set(eval_indices)) == 0
    assert len(set(training_indices) & set(test_indices)) == 0
    assert len(set(eval_indices) & set(test_indices)) == 0
    
    # Create the datasets
    train_data = data[training_indices]
    eval_data = data[eval_indices]
    test_data = data[test_indices]
    
    if write_to_file:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, '../../data/few_shot')
        
        os.makedirs(full_path, exist_ok=True)  # Ensure the directory exists

        with open(f'{full_path}/train_{num_shots}_shot.json', 'w') as train_file:
            json.dump(train_data.tolist(), train_file, indent=4)
        
        with open(f'{full_path}/eval_{num_shots}_shot.json', 'w') as eval_file:
            json.dump(eval_data.tolist(), eval_file, indent=4)
        
        with open(f'{full_path}/test_{num_shots}_shot.json', 'w') as test_file:
            json.dump(test_data.tolist(), test_file, indent=4)
    
    return train_data.tolist(), eval_data.tolist(), test_data.tolist()

# Example usage
if __name__ == "__main__":
    # Load your dataset
    with open('../../Data/MBPP_transformed_code_examples/sanitized-MBPP-midio.json', 'r') as file:
        dataset = json.load(file)
    
    num_shots = 10
    eval_size_percentage = 0.2  # 20% of possible samples after making the training set
    
    train_data, eval_data, test_data = split_on_shots(num_shots, eval_size_percentage, dataset, write_to_file=True)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Evaluation samples: {len(eval_data)}")
    print(f"Test samples: {len(test_data)}")