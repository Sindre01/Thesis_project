import json
import os
import numpy as np
import random
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# split dataset into training, validation, and test sets
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

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(validation_data)}")
    print(f"Test set size: {len(test_data)}")
    return train_data.tolist(), validation_data.tolist(), test_data.tolist()
    # print("\nLabels only in training dataset:")
    # for label in sorted(labels_only_in_training):
    #     print(label)

#Splits dataset for few-shot prompting, where num-shots is number of samples to be included in train-set
def split_on_shots_nodes(num_shots, desired_val_size, dataset, seed=62, random_few_shots=False, write_to_file=False):
    """
    Split the dataset into training, validation, and test sets for few-shot learning.
    - For each sample, the combined labels of library functions and textual instance types are used.
    - The training set is built by selecting num-shot samples that cover as many labels as possible,
      unless random=True, in which case num_shots are randomly selected.
    - The remaining samples are split into validation and test sets based on desired_val_size.
    - Samples that only have labels from the training set are included in the validation and test sets.
    - ***If not enough samples are available, all remaining samples are assigned to the validation or test set.
    - The resulting datasets are written to files if write_to_file is True.

    Tries to split into validation and test sets 50/50 (desired_val_size = 0.5), but because of 
    MultiLabelStratifiedShuffleSplit, the actual split may vary. 
    
    This is because:
    1. **Stratification Constraints**:
        - The `MultilabelStratifiedShuffleSplit` prioritizes maintaining the relative frequency of labels across 
        validation and test sets over adhering strictly to the requested split proportions.
        - For example, if certain labels are rare, they may need to be placed entirely in one set to preserve 
        label distributions.

    2. **Multilabel Data**:
        - Unlike single-label data, where each sample belongs to only one class, multilabel data assigns each 
        sample to multiple classes. This increases the complexity of stratification and may lead to deviations 
        from the exact split sizes.

    3. **Small or Imbalanced Dataset**:
        - When the dataset is small or highly imbalanced, the stratification process may reassign samples 
        between validation and test sets to ensure all labels are represented adequately.

    4. **Minimum Label Constraints**:
        - Stratification requires that each label appears at least twice in the dataset to be split. If a label 
        is rare (e.g., appears only once), it may cause adjustments in the split sizes.

    As a result, while `desired_val_size` specifies the intended proportion for the validation set, the actual 
    sizes of the validation and test sets may differ slightly. These deviations ensure that the stratification 
    process produces meaningful and balanced splits for multilabel classification tasks.
    """

    # If num_shots == 0, fallback to a different split logic
    if num_shots == 0:
        print("NOT WORKING: Number of shots is zero. splitting into only validation and test sets")
        return split(dataset, write_to_file, 0, 0.2, 0.8)  # Some custom function or logic not shown

    # Set seeds for reproducibility
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    
    data = np.array(dataset, dtype=object)
    
    # Clean up and combine labels
    combined_labels_list = []
    for item in data:
        library_functions = item.get('library_functions', [])
        # Remove 'root.std.' if present
        library_functions = [func.replace('root.std.', '') for func in library_functions]
        item['library_functions'] = library_functions
    
        visual_node_types = item.get('visual_node_types', [])
        item['visual_node_types'] = visual_node_types
    
        combined_labels = library_functions + visual_node_types
        combined_labels_list.append(combined_labels)
    
    # Binarize labels
    mlb = MultiLabelBinarizer()
    labels_array = mlb.fit_transform(combined_labels_list)
    all_indices = np.arange(len(data))
    
    # Build the training set with num_shots samples
    if random_few_shots:
        # -------------------
        # PICK RANDOM SAMPLES
        # -------------------
        print("Random selection of training samples.")
        # Ensure we don't pick more than available
        num_shots_actual = min(num_shots, len(all_indices))
        shuffled_indices = list(all_indices)
        np.random.shuffle(shuffled_indices)
        training_indices = shuffled_indices[:num_shots_actual]
    else:
        # ------------------------------------------------------------------------
        # ORIGINAL LOGIC: Select training samples that cover as many labels as possible
        # ------------------------------------------------------------------------
        print("Coverage-based selection of training samples.")
        training_indices = []
        label_coverage = set()
        
        sample_labels = []
        for idx, labels in enumerate(combined_labels_list):
            sample_labels.append({
                'index': idx,
                'labels': set(labels),
                'num_labels': len(set(labels))
            })
        
        # Sort samples by number of labels (descending)
        sample_labels.sort(key=lambda x: x['num_labels'], reverse=True)
        
        for sample in sample_labels:
            if len(training_indices) >= num_shots:
                break
            # Only pick this sample if it adds at least one new label
            if not sample['labels'].issubset(label_coverage):
                training_indices.append(sample['index'])
                label_coverage.update(sample['labels'])
        
        # If not enough unique coverage samples found, fill randomly
        if len(training_indices) < num_shots:
            print(f"Only {len(training_indices)} samples found with unique labels. Filling the rest randomly.")
            remaining_indices = list(set(all_indices) - set(training_indices))
            random.shuffle(remaining_indices)
            training_indices.extend(remaining_indices[:num_shots - len(training_indices)])
    # ------------------- END OF TRAINING SET SELECTION -------------------
    
    # Collect labels used in the training set
    training_labels = set(label for idx in training_indices for label in combined_labels_list[idx])
    
    # Remaining indices
    remaining_indices = list(set(all_indices) - set(training_indices))
    
    # Filter samples that only have labels from the training set
    filtered_indices = [
        idx for idx in remaining_indices
        if set(combined_labels_list[idx]).issubset(training_labels)
    ]
    
    # Number of possible samples after filtering
    total_possible_samples = len(filtered_indices)
    
    # Calculate the number of validation samples based on desired_val_size
    num_val_samples = int(desired_val_size * total_possible_samples)
    num_val_samples = max(num_val_samples, 1)  # Ensure at least one sample
    num_test_samples = total_possible_samples - num_val_samples
    
    print(f"Total available samples for evaluation with nodes covered in few-shot dataset (training set): {total_possible_samples}")
    print(f"> Samples possible for validation: {num_val_samples}")
    print(f"> Samples possible for testing: {num_test_samples}")
    
    # Shuffle filtered indices to randomize before splitting
    random.shuffle(filtered_indices)
    
    # If there are enough samples to split
    if num_val_samples > 0 and num_test_samples > 0:
        # Prepare data and labels for stratified splitting
        filtered_data = data[filtered_indices]
        filtered_labels = labels_array[filtered_indices]
        
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            train_size=float(num_val_samples) / total_possible_samples,
            test_size=float(num_test_samples) / total_possible_samples,
            random_state=SEED
        )
        val_indices_rel, test_indices_rel = next(msss.split(filtered_data, filtered_labels))
        
        val_indices = [filtered_indices[i] for i in val_indices_rel]
        test_indices = [filtered_indices[i] for i in test_indices_rel]
    else:
        # Not enough samples to create both sets properly
        print("Not enough samples for validation and test sets. Assigning all to validation set.")
        val_indices = filtered_indices[:num_val_samples]
        test_indices = filtered_indices[num_val_samples:]
    
    # Ensure no overlap
    assert len(set(training_indices) & set(val_indices)) == 0
    assert len(set(training_indices) & set(test_indices)) == 0
    assert len(set(val_indices) & set(test_indices)) == 0
    
    # Create the datasets
    train_data = data[training_indices]
    val_data = data[val_indices]
    test_data = data[test_indices]
    
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Optionally write to file
    if write_to_file:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, '../../data/few_shot')
        
        os.makedirs(full_path, exist_ok=True)
        
        with open(f'{full_path}/train_{num_shots}_shot.json', 'w') as train_file:
            json.dump(train_data.tolist(), train_file, indent=4)
        
        with open(f'{full_path}/validation_{num_shots}_shot.json', 'w') as val_file:
            json.dump(val_data.tolist(), val_file, indent=4)
        
        with open(f'{full_path}/test_{num_shots}_shot.json', 'w') as test_file:
            json.dump(test_data.tolist(), test_file, indent=4)
    
    return train_data.tolist(), val_data.tolist(), test_data.tolist()


def read_dataset_to_json(file_path):
    with open(file_path, 'r') as file:
        dataset = json.load(file)
    return dataset



    import os
import json
import numpy as np
import random
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def multi_shot_split_ensure_valtest_coverage(
    num_shots_list,
    desired_val_size,
    dataset,
    seed=62,
    random_pick=False,
    write_to_file=False
):
    """
    1) Splits 'dataset' into a single val/test with fixed coverage for all runs.
    2) For each 'num_shots' in num_shots_list, picks a training set that covers
       all labels found in val/test (L_valtest). If fewer than num_shots samples 
       are needed for coverage, we fill extra slots from the leftover pool.
    3) Returns a dict of {shot_size: train_set}, plus the single val/test sets.

    Args:
        num_shots_list (list/tuple[int]): e.g. [5, 10]
        desired_val_size (float): fraction (0 < desired_val_size < 0.5 ideally)
        dataset (list[dict]): your entire data
        seed (int): random seed
        random_pick (bool): 
            If True, after ensuring coverage of L_valtest, fill extra with random.
            If False, try to pick coverage-based (sort by #labels desc), then fill random if needed.
        write_to_file (bool): if True, write splits to disk (paths below).

    Returns:
        train_sets_dict (dict): { num_shots: <list of training samples> }
        val_data (list): the single fixed validation set
        test_data (list): the single fixed test set
    """

    # ----------------------------------------------------------------
    # 0. Basic Setup
    # ----------------------------------------------------------------
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)

    data = np.array(dataset, dtype=object)
    
    # Combine labels
    combined_labels_list = []
    for item in data:
        # Clean library_functions
        lib_funcs = item.get('library_functions', [])
        lib_funcs = [lf.replace('root.std.', '') for lf in lib_funcs]
        item['library_functions'] = lib_funcs

        # Visual node types
        visual_nodes = item.get('visual_node_types', [])
        item['visual_node_types'] = visual_nodes

        combined_labels_list.append(lib_funcs + visual_nodes)

    # Binarize
    mlb = MultiLabelBinarizer()
    labels_array = mlb.fit_transform(combined_labels_list)
    all_indices = np.arange(len(data))

    # ----------------------------------------------------------------
    # 1. Single Pass for Validation & Test (Fixed for all shot sizes)
    # ----------------------------------------------------------------

    # We'll do a two-stage split:
    # 1) Separate (val+test) vs. train-candidates
    # 2) Within (val+test), separate val from test ~ 50/50
    #
    # For example, if desired_val_size=0.2, we do:
    #  => total_eval_size = 0.4 (20% val, 20% test)
    #  => leftover for training = 0.6

    total_eval_size = 2 * desired_val_size
    if total_eval_size >= 1.0:
        raise ValueError(
            "desired_val_size is too large => not enough data left for training."
        )

    leftover_fraction = 1.0 - total_eval_size
    msss_1 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=leftover_fraction,
        test_size=total_eval_size,
        random_state=SEED
    )
    train_candidates_indices, eval_indices = next(msss_1.split(data, labels_array))

    # Now split eval_indices into val/test
    eval_data = data[eval_indices]
    eval_labels = labels_array[eval_indices]

    msss_2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=0.5,  # half of eval => val
        test_size=0.5,   # half of eval => test
        random_state=SEED
    )
    val_rel, test_rel = next(msss_2.split(eval_data, eval_labels))
    val_indices = eval_indices[val_rel]
    test_indices = eval_indices[test_rel]

    val_data = data[val_indices]
    test_data = data[test_indices]

    # ----------------------------------------------------------------
    # 2. Determine the Label Coverage of val/test (L_valtest)
    # ----------------------------------------------------------------
    valtest_labels = set()
    for idx in val_indices:
        valtest_labels.update(combined_labels_list[idx])
    for idx in test_indices:
        valtest_labels.update(combined_labels_list[idx])

    # The leftover pool from which we'll pick training sets
    leftover_indices = np.array(train_candidates_indices)

    # Pre-compute coverage info
    coverage_info = []
    for idx in leftover_indices:
        coverage_info.append({
            'index': idx,
            'labels': set(combined_labels_list[idx]),
            'num_labels': len(set(combined_labels_list[idx]))
        })

    # Sort descending by number of labels (helpful for coverage-based selection)
    coverage_info_sorted = sorted(coverage_info, key=lambda x: x['num_labels'], reverse=True)

    # ----------------------------------------------------------------
    # 3. Build Training Sets for Each Shot Value
    # ----------------------------------------------------------------
    train_sets_dict = {}

    for shots in num_shots_list:
        # 3.1 We must ensure coverage of all val/test labels
        #     We'll pick whichever leftover samples help us cover these labels first.
        needed_labels = set(valtest_labels)  # copy of the val/test label set
        chosen_indices = []

        # coverage-based picking from coverage_info_sorted
        if not random_pick:
            for sample in coverage_info_sorted:
                if len(needed_labels) == 0:
                    break
                # If this sample has overlap with what's still needed, pick it
                if len(sample['labels'] & needed_labels) > 0:
                    chosen_indices.append(sample['index'])
                    needed_labels -= sample['labels']  # remove covered labels

            # If we still haven't covered everything but ran out of coverage_info_sorted,
            # it means it's impossible to cover all val/test labels with leftover data.
            # We'll just proceed with what we have. (Or raise an error if you want.)
            if len(needed_labels) > 0:
                print(
                    f"Warning: Not all val/test labels can be covered for {shots}-shot. "
                    f"Uncovered labels: {needed_labels}"
                )

            # If we have covered everything but haven't reached 'shots' samples, fill the remainder.
            if len(chosen_indices) < shots:
                leftover_pool = list(set(leftover_indices) - set(chosen_indices))
                random.shuffle(leftover_pool)
                # Fill up to 'shots'
                needed = shots - len(chosen_indices)
                chosen_indices.extend(leftover_pool[:needed])

        else:
            # random_pick == True
            # We still want to ensure coverage, but we do it in a simpler random-based approach:
            # 1) Shuffle leftover
            # 2) Keep adding samples if they help cover needed_labels
            # 3) If coverage is done but we haven't reached shots, fill randomly
            leftover_pool = list(leftover_indices)
            random.shuffle(leftover_pool)

            for idx_ in leftover_pool:
                if len(needed_labels) == 0:
                    break
                labels_here = set(combined_labels_list[idx_])
                if len(labels_here & needed_labels) > 0:
                    chosen_indices.append(idx_)
                    needed_labels -= labels_here

            # If coverage incomplete
            if len(needed_labels) > 0:
                print(
                    f"Warning: Not all val/test labels can be covered for {shots}-shot. "
                    f"Uncovered labels: {needed_labels}"
                )

            # Fill up extra if we haven't reached shots
            if len(chosen_indices) < shots:
                leftover_pool_remaining = list(set(leftover_pool) - set(chosen_indices))
                random.shuffle(leftover_pool_remaining)
                needed = shots - len(chosen_indices)
                chosen_indices.extend(leftover_pool_remaining[:needed])

        # 3.2 Build final train set for this shot
        train_data_for_shot = data[chosen_indices]
        train_sets_dict[shots] = train_data_for_shot.tolist()

    # ----------------------------------------------------------------
    # 4. (Optional) Write to File
    # ----------------------------------------------------------------
    if write_to_file:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, '../../data/few_shot')
        os.makedirs(full_path, exist_ok=True)

        # Save val/test once (fixed)
        with open(os.path.join(full_path, 'val_fixed.json'), 'w') as vf:
            json.dump(val_data.tolist(), vf, indent=4)
        with open(os.path.join(full_path, 'test_fixed.json'), 'w') as tf:
            json.dump(test_data.tolist(), tf, indent=4)

        # Save each training set
        for shot_size, train_list in train_sets_dict.items():
            with open(os.path.join(full_path, f'train_{shot_size}_shot.json'), 'w') as trainf:
                json.dump(train_list, trainf, indent=4)

    # Return the dictionary of training sets + the single val/test
    return train_sets_dict, val_data.tolist(), test_data.tolist()


import numpy as np
import random
from sklearn.model_selection import KFold

def kfold_few_shot(
    dataset,
    n_splits=5,
    seed=42,
    num_shots_list=(5, 10)
):
    """
    Perform k-fold cross validation with few-shot subsets (e.g., 5-shot, 10-shot),
    without requiring coverage of all validation labels.
    
    Args:
        dataset (list): Each element is a data sample (dict, or any format).
        n_splits (int): Number of folds for k-fold cross validation.
        seed (int): Random seed.
        num_shots_list (tuple): e.g. (5, 10), the shot sizes we want to sample.

    Returns:
        results (dict):
            {
              'fold_0': {
                  'val_indices': [...],
                  'train_5_indices': [...],
                  'train_10_indices': [...]
              },
              'fold_1': {...},
              ...
            }
        Each fold stores:
          - The validation indices
          - The chosen training indices for 5-shot
          - The chosen training indices for 10-shot

    Note: You can then retrieve the actual data by indexing into `dataset`.
    """
    random.seed(seed)
    np.random.seed(seed)

    data_array = np.array(dataset, dtype=object)
    n_samples = len(data_array)

    # Create KFold object (shuffle=True for random splitting)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results = {}
    fold_index = 0

    for train_indices_full, val_indices in kf.split(data_array):
        # train_indices_full is the combined set of (k-1) folds
        # val_indices is the remaining 1 fold
        
        # We'll pick few-shot subsets from train_indices_full
        train_pool = list(train_indices_full)
        random.shuffle(train_pool)  # shuffle to ensure random selection

        # For each shot size, pick that many examples from train_pool
        fold_result = {'val_indices': val_indices.tolist()}

        for shots in num_shots_list:
            # Ensure we don't exceed what's available
            chosen_count = min(shots, len(train_pool))
            train_shot_indices = train_pool[:chosen_count]
            
            # Optionally, if you want a separate random draw for each shot size,
            # you could shuffle again or slice a different portion.
            # But here, we'll just slice from the front. 
            # If you want them truly independent, you'd do something like:
            #   random.shuffle(train_pool)
            #   train_shot_indices = train_pool[:chosen_count]

            # Store
            fold_result[f'train_{shots}_indices'] = train_shot_indices

        results[f'fold_{fold_index}'] = fold_result
        fold_index += 1

    return results

def split(dataset, write_to_file = False, desired_train_size = 0.7, desired_val_size = 0.2, desired_test_size= 0.1, seed=62):
    pass

