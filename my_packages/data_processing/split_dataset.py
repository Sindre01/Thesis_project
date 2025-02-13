import json
import os
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# #Splits dataset for few-shot prompting, where num-shots is number of samples to be included in train-set
# def split_on_shots_nodes(num_shots, desired_val_size, dataset, seed=62, random_few_shots=False, write_to_file=False):
#     """
#     Split the dataset into training, validation, and test sets for few-shot learning.
#     - For each sample, the combined labels of external functions and textual instance types are used.
#     - The training set is built by selecting num-shot samples that cover as many labels as possible,
#       unless random=True, in which case num_shots are randomly selected.
#     - The remaining samples are split into validation and test sets based on desired_val_size.
#     - Samples that only have labels from the training set are included in the validation and test sets.
#     - ***If not enough samples are available, all remaining samples are assigned to the validation or test set.
#     - The resulting datasets are written to files if write_to_file is True.

#     Tries to split into validation and test sets 50/50 (desired_val_size = 0.5), but because of 
#     MultiLabelStratifiedShuffleSplit, the actual split may vary. 
    
#     This is because:
#     1. **Stratification Constraints**:
#         - The `MultilabelStratifiedShuffleSplit` prioritizes maintaining the relative frequency of labels across 
#         validation and test sets over adhering strictly to the requested split proportions.
#         - For example, if certain labels are rare, they may need to be placed entirely in one set to preserve 
#         label distributions.

#     2. **Multilabel Data**:
#         - Unlike single-label data, where each sample belongs to only one class, multilabel data assigns each 
#         sample to multiple classes. This increases the complexity of stratification and may lead to deviations 
#         from the exact split sizes.

#     3. **Small or Imbalanced Dataset**:
#         - When the dataset is small or highly imbalanced, the stratification process may reassign samples 
#         between validation and test sets to ensure all labels are represented adequately.

#     4. **Minimum Label Constraints**:
#         - Stratification requires that each label appears at least twice in the dataset to be split. If a label 
#         is rare (e.g., appears only once), it may cause adjustments in the split sizes.

#     As a result, while `desired_val_size` specifies the intended proportion for the validation set, the actual 
#     sizes of the validation and test sets may differ slightly. These deviations ensure that the stratification 
#     process produces meaningful and balanced splits for multilabel classification tasks.
#     """

#     # If num_shots == 0, fallback to a different split logic
#     if num_shots == 0:
#         print("NOT WORKING: Number of shots is zero.")
#         #return split(dataset, write_to_file, 0, 0.2, 0.8)  # Some custom function or logic not shown

#     # Set seeds for reproducibility
#     SEED = seed
#     random.seed(SEED)
#     np.random.seed(SEED)
    
#     data = np.array(dataset, dtype=object)
    
#     # Clean up and combine labels
#     combined_labels_list = []
#     for item in data:
#         external_functions = item.get('external_functions', [])
#         # Remove 'root.std.' if present
#         external_functions = [func.replace('root.std.', '') for func in external_functions]
#         item['external_functions'] = external_functions
    
#         visual_node_types = item.get('visual_node_types', [])
#         item['visual_node_types'] = visual_node_types
    
#         combined_labels = external_functions + visual_node_types
#         combined_labels_list.append(combined_labels)
    
#     # Binarize labels
#     mlb = MultiLabelBinarizer()
#     labels_array = mlb.fit_transform(combined_labels_list)
#     all_indices = np.arange(len(data))
    
#     # Build the training set with num_shots samples
#     if random_few_shots:
#         # -------------------
#         # PICK RANDOM SAMPLES
#         # -------------------
#         print("Random selection of training samples.")
#         # Ensure we don't pick more than available
#         num_shots_actual = min(num_shots, len(all_indices))
#         shuffled_indices = list(all_indices)
#         np.random.shuffle(shuffled_indices)
#         training_indices = shuffled_indices[:num_shots_actual]
#     else:
#         # ------------------------------------------------------------------------
#         # ORIGINAL LOGIC: Select training samples that cover as many labels as possible
#         # ------------------------------------------------------------------------
#         print("Coverage-based selection of training samples.")
#         training_indices = []
#         label_coverage = set()
        
#         sample_labels = []
#         for idx, labels in enumerate(combined_labels_list):
#             sample_labels.append({
#                 'index': idx,
#                 'labels': set(labels),
#                 'num_labels': len(set(labels))
#             })
        
#         # Sort samples by number of labels (descending)
#         sample_labels.sort(key=lambda x: x['num_labels'], reverse=True)
        
#         for sample in sample_labels:
#             if len(training_indices) >= num_shots:
#                 break
#             # Only pick this sample if it adds at least one new label
#             if not sample['labels'].issubset(label_coverage):
#                 training_indices.append(sample['index'])
#                 label_coverage.update(sample['labels'])
        
#         # If not enough unique coverage samples found, fill randomly
#         if len(training_indices) < num_shots:
#             print(f"Only {len(training_indices)} samples found with unique labels. Filling the rest randomly.")
#             remaining_indices = list(set(all_indices) - set(training_indices))
#             random.shuffle(remaining_indices)
#             training_indices.extend(remaining_indices[:num_shots - len(training_indices)])
#     # ------------------- END OF TRAINING SET SELECTION -------------------
    
#     # Collect labels used in the training set
#     training_labels = set(label for idx in training_indices for label in combined_labels_list[idx])
    
#     # Remaining indices
#     remaining_indices = list(set(all_indices) - set(training_indices))
    
#     # Filter samples that only have labels from the training set
#     filtered_indices = [
#         idx for idx in remaining_indices
#         if set(combined_labels_list[idx]).issubset(training_labels)
#     ]
    
#     # Number of possible samples after filtering
#     total_possible_samples = len(filtered_indices)
    
#     # Calculate the number of validation samples based on desired_val_size
#     num_val_samples = int(desired_val_size * total_possible_samples)
#     num_val_samples = max(num_val_samples, 1)  # Ensure at least one sample
#     num_test_samples = total_possible_samples - num_val_samples
    
#     print(f"Total available samples for evaluation with nodes covered in few-shot dataset (training set): {total_possible_samples}")
#     print(f"> Samples possible for validation: {num_val_samples}")
#     print(f"> Samples possible for testing: {num_test_samples}")
    
#     # Shuffle filtered indices to randomize before splitting
#     random.shuffle(filtered_indices)
    
#     # If there are enough samples to split
#     if num_val_samples > 0 and num_test_samples > 0:
#         # Prepare data and labels for stratified splitting
#         filtered_data = data[filtered_indices]
#         filtered_labels = labels_array[filtered_indices]
        
#         msss = MultilabelStratifiedShuffleSplit(
#             n_splits=1,
#             train_size=float(num_val_samples) / total_possible_samples,
#             test_size=float(num_test_samples) / total_possible_samples,
#             random_state=SEED
#         )
#         val_indices_rel, test_indices_rel = next(msss.split(filtered_data, filtered_labels))
        
#         val_indices = [filtered_indices[i] for i in val_indices_rel]
#         test_indices = [filtered_indices[i] for i in test_indices_rel]
#     else:
#         # Not enough samples to create both sets properly
#         print("Not enough samples for validation and test sets. Assigning all to validation set.")
#         val_indices = filtered_indices[:num_val_samples]
#         test_indices = filtered_indices[num_val_samples:]
    
#     # Ensure no overlap
#     assert len(set(training_indices) & set(val_indices)) == 0
#     assert len(set(training_indices) & set(test_indices)) == 0
#     assert len(set(val_indices) & set(test_indices)) == 0
    
#     # Create the datasets
#     train_data = data[training_indices]
#     val_data = data[val_indices]
#     test_data = data[test_indices]
    
#     print(f"Training set size: {len(train_data)}")
#     print(f"Validation set size: {len(val_data)}")
#     print(f"Test set size: {len(test_data)}")
    
#     # Optionally write to file
#     if write_to_file:
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         full_path = os.path.join(current_dir, '../../data/few_shot')
        
#         os.makedirs(full_path, exist_ok=True)
        
#         with open(f'{full_path}/train_{num_shots}_shot.json', 'w') as train_file:
#             json.dump(train_data.tolist(), train_file, indent=4)
        
#         with open(f'{full_path}/validation_{num_shots}_shot.json', 'w') as val_file:
#             json.dump(val_data.tolist(), val_file, indent=4)
        
#         with open(f'{full_path}/test_{num_shots}_shot.json', 'w') as test_file:
#             json.dump(test_data.tolist(), test_file, indent=4)
    
#     return train_data.tolist(), val_data.tolist(), test_data.tolist()

  
def multi_stratified_split(dataset, write_to_file, eval_size, seed=58):
    SEED = seed
    prompts = [item['prompts'][0] for item in dataset]   # e.g., take the 'prompts' list as your features
    responses = [item['external_functions']  for item in dataset]  # e.g., take the ?? as your labels

    # Convert to numpy arrays (optional, but often convenient for sklearn usage)
    prompts = np.array(prompts, dtype=object)
    responses = np.array(responses, dtype=object)
    #Convert to a numpy array of objects
    data = np.array(dataset, dtype=object)
    # 'responses' is a list of lists of external functions
    responses_raw = [item['external_functions'] for item in dataset]

    # Build a list of all unique external functions
    all_libs = sorted({fn for libs in responses_raw for fn in libs})
    lib_to_idx = {fn: i for i, fn in enumerate(all_libs)}
    num_labels = len(all_libs)

    # Create a binary indicator matrix Y of shape [n_samples, num_labels]
    Y = np.zeros((len(responses_raw), num_labels), dtype=int)
    for i, libs in enumerate(responses_raw):
        for fn in libs:
            Y[i, lib_to_idx[fn]] = 1

    
    # We want to do an 40% test split, preserving distribution of multi-labels
    mskf = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=eval_size, random_state=SEED)

    # 'data' is your full dataset, 'Y' is the multi-label matrix
    for train_index, test_index in mskf.split(data, Y):
        train = data[train_index]
        test  = data[test_index]
        
        Y_train = Y[train_index]
        Y_test  = Y[test_index]

    print("Train shape:", train.shape)
    print("Test  shape:", test.shape)
    # Now we do a second split from the 'test' portion to get 'val' and final 'test'
    # so each is half of that portion (20% / 20% of original data),
    # but still preserving multi-label distribution.

    mskf2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    for val_index, test_index_2 in mskf2.split(test, Y_test):
        val  = test[val_index]
        test = test[test_index_2]
        
        Y_val  = Y_test[val_index]
        Y_test = Y_test[test_index_2]

    print("Final shapes:")
    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("Test:", test.shape)

    def collect_external_functions(data_subset):
        """
        Collects all external functions from the 'external_functions' key
        in a given subset of the dataset and returns them as a set.
        """
        libs = set()
        for item in data_subset:
            for fn in item['external_functions']:
                libs.add(fn)
        return libs

    train_libs = collect_external_functions(train)
    val_libs   = collect_external_functions(val)
    test_libs  = collect_external_functions(test)
    print("Number of unique external_functions in train:", len(train_libs))
    print("Number of unique external_functions in validation:", len(val_libs))
    print("Number of unique external_functions in test:", len(test_libs))

    # Step 2: Identify external functions present in val or test but not in train.

    val_extra_libs  = val_libs - train_libs
    test_extra_libs = test_libs - train_libs

    # Step 3: Print the results.

    if val_extra_libs:
        print("external_functions in validation set not present in training set:")
        print(val_extra_libs)
    else:
        print("No 'new' external_functions in validation set that aren't in training.")

    if test_extra_libs:
        print("external_functions in test set not present in training set:")
        print(test_extra_libs)
    else:
        print("No 'new' external_functions in test set that aren't in training.")

    #Write to file:
    if write_to_file:
        full_path = os.path.join('../../data')

        with open(f'{full_path}/train_dataset.json', 'w') as train_file:
            json.dump(train.tolist(), train_file, indent=4)

        with open(f'{full_path}/validation_dataset.json', 'w') as eval_file:
            json.dump(val.tolist(), eval_file, indent=4)

        with open(f'{full_path}/test_dataset.json', 'w') as test_file:
            json.dump(test.tolist(), test_file, indent=4)

    return train, val, test



def random_split(dataset, write_to_file, eval_size, seed=42):
    data = np.array(dataset, dtype=object)
    train, test= train_test_split(
        data, test_size=eval_size, random_state=seed
    )
    print("Shapes:")
    print("train:", train.shape)
    print("test:", test.shape)

    val, test = train_test_split(
        test, test_size=0.5, random_state=seed, shuffle=True, 
    )

    print("Shapes:")
    print("train:", train.shape)
    print("validation:", val.shape)
    print("test:", test.shape)

    if write_to_file:
        full_path = os.path.join('../../data')

        with open(f'{full_path}/train_dataset.json', 'w') as train_file:
            json.dump(train.tolist(), train_file, indent=4)

        with open(f'{full_path}/validation_dataset.json', 'w') as eval_file:
            json.dump(val.tolist(), eval_file, indent=4)

        with open(f'{full_path}/test_dataset.json', 'w') as test_file:
            json.dump(test.tolist(), test_file, indent=4)