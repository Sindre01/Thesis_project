import json
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

def get_stratified_kfold_splits(dataset: list[dict], k_folds=5):
    """
    Splits dataset into stratified k-folds based on function category.
    Stratification happens **across** folds to maintain class distribution.
    """
    function_categories = [data["external_functions"] for data in dataset]  # Stratify by function type

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    folds = []

    for train_idx, test_idx in skf.split(dataset, function_categories):
        train_data = [dataset[i] for i in train_idx]
        test_data = [dataset[i] for i in test_idx]

        # Verify stratification across folds
        print(f"Fold {len(folds)+1}: Train={len(train_data)}, Test={len(test_data)}")
        print("Train Distribution:", Counter([d["external_functions"] for d in train_data]))
        print("Test Distribution:", Counter([d["external_functions"] for d in test_data]), "\n")

        folds.append((train_data, test_data))

    
    return folds

def create_kfold_splits(dataset: list[dict], k_folds=5, write_to_file=False):
    """Splits dataset into k folds."""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    folds = [(train_idx, test_idx) for train_idx, test_idx in kf.split(dataset)]
    

    #Write to file:
    if write_to_file:
        full_path = os.path.join(f'{project_root}/data/MBPP_Midio_50/splits/k_fold')
        
        for fold in range(k_folds):
            train_data = [dataset[i] for i in folds[fold][0]]
            test_data = [dataset[i] for i in folds[fold][1]]

            with open(f'{full_path}/train_dataset_{fold}.json', 'w') as train_file:
                json.dump(train_data, train_file, indent=4)

            with open(f'{full_path}/test_dataset_{fold}.json', 'w') as test_file:
                json.dump(test_data, test_file, indent=4)

    return dataset, folds

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
        full_path = os.path.join('../../data/MBPP_Midio_50/splits')

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
        full_path = os.path.join('../../data/MBPP_Midio_50/splits')

        with open(f'{full_path}/train_dataset.json', 'w') as train_file:
            json.dump(train.tolist(), train_file, indent=4)

        with open(f'{full_path}/validation_dataset.json', 'w') as eval_file:
            json.dump(val.tolist(), eval_file, indent=4)

        with open(f'{full_path}/test_dataset.json', 'w') as test_file:
            json.dump(test.tolist(), test_file, indent=4)