
from Scripts.install_dependencies import install_dependencies
install_dependencies()

import json
import numpy as np
import random
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from Scripts.Analysis.analyze_datasets import (
    analyze_instance_distribution,
    analyze_library_distribution,
    analyze_visual_node_types_distribution
)

def split_dataset_on_shots(num_shots, dataset_path):
    # Set the number of shots for the training set size
    num_shots = 5
    
    # Set seeds for reproducibility
    SEED = 62
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Desired split ratio
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    
    # Load the dataset
    with open("../../Data/MBPP_transformed_code_examples/sanitized-MBPP-midio.json", 'r') as file:
        data = json.load(file)
    
    data = np.array(data)
    
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
    
    training_labels = set(label for idx in training_indices for label in combined_labels_list[idx])
    remaining_indices = list(set(all_indices) - set(training_indices))
    
    # Filter samples that only have labels from the training set
    filtered_indices = [idx for idx in remaining_indices
                        if set(combined_labels_list[idx]).issubset(training_labels)]
    
    # Compute desired exact counts for val and test
    # With num_shots=10 (70%), total is approx 14. So val ~3, test ~1
    desired_val = int(round((num_shots / train_ratio) * val_ratio))
    desired_test = int(round((num_shots / train_ratio) * test_ratio))
    
    # Ensure at least 1 sample if we intended to have test set
    desired_val = max(desired_val, 1)
    desired_test = max(desired_test, 1)
    
    # If rounding causes mismatch, adjust
    approx_total = int(round(num_shots / train_ratio))
    allocated = num_shots + desired_val + desired_test
    if allocated < approx_total:
        # Add leftover to validation
        difference = approx_total - allocated
        desired_val += difference
    
    # Now we have desired_val and desired_test. Check feasibility:
    actual_filtered = len(filtered_indices)
    total_needed = desired_val + desired_test
    
    if actual_filtered < total_needed:
        # Not enough filtered samples to match desired counts
        if actual_filtered == 0:
            val_count = 0
            test_count = 0
        elif actual_filtered == 1:
            val_count = 1
            test_count = 0
        else:
            val_count = max(int(round(actual_filtered * 2/3)), 1)
            test_count = actual_filtered - val_count
            if desired_test > 0 and test_count == 0 and val_count > 1:
                val_count -= 1
                test_count = 1
    else:
        # More filtered samples than needed: trim them
        random.shuffle(filtered_indices)
        filtered_indices = filtered_indices[:total_needed]
        val_count = desired_val
        test_count = desired_test
    
    filtered_data = data[filtered_indices]
    filtered_labels = labels_array[filtered_indices]
    
    validation_indices = []
    test_indices = []
    
    # Perform stratified split if possible and we have at least some test samples
    if val_count > 0 and test_count > 0 and len(filtered_indices) >= (val_count + test_count):
        try:
            msss = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                train_size=val_count,
                test_size=test_count,
                random_state=42
            )
            val_indices_rel, test_indices_rel = next(msss.split(filtered_data, filtered_labels))
            validation_indices = [filtered_indices[i] for i in val_indices_rel]
            test_indices = [filtered_indices[i] for i in test_indices_rel]
        except:
            random.shuffle(filtered_indices)
            validation_indices = filtered_indices[:val_count]
            test_indices = filtered_indices[val_count:val_count + test_count]
    else:
        # No test or not enough samples, all go to validation
        validation_indices = filtered_indices
        test_indices = []
    
    # Ensure uniqueness
    assert len(set(training_indices) & set(validation_indices)) == 0
    assert len(set(training_indices) & set(test_indices)) == 0
    assert len(set(validation_indices) & set(test_indices)) == 0
    
    train_data = data[training_indices]
    validation_data = data[validation_indices]
    test_data = data[test_indices]
    
    with open(f'../../Data/Few-shot/train_{num_shots}_shot.json', 'w') as train_file:
        json.dump(train_data.tolist(), train_file, indent=4)
    
    with open(f'../../Data/Few-shot/validation_{num_shots}_shot.json', 'w') as eval_file:
        json.dump(validation_data.tolist(), eval_file, indent=4)
    
    with open(f'../../Data/Few-shot/test_{num_shots}_shot.json', 'w') as test_file:
        json.dump(test_data.tolist(), test_file, indent=4)
    
    total_samples_actual = len(train_data) + len(validation_data) + len(test_data)
    print(f"Total samples: {total_samples_actual}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(validation_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Analyze distributions
    analyze_library_distribution(train_data, validation_data, test_data)
    analyze_instance_distribution(train_data, validation_data, test_data)
    analyze_visual_node_types_distribution(train_data, validation_data, test_data)

    return train_data, validation_data, test_data
