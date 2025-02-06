import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def get_label_distribution(data_subset, label_type):
    subset_labels = []
    for item in data_subset:
        labels = item.get(label_type, [])
        #remove "root.std" from the labels string
        labels = [label.replace("root.std.", "") for label in labels]
        subset_labels.extend(labels)
    return subset_labels

def analyze_visual_node_types_distribution(train_data, validation_data, test_data):
    train_library_distribution = Counter(get_label_distribution(train_data, 'visual_node_types'))
    validation_library_distribution = Counter(get_label_distribution(validation_data, 'visual_node_types'))
    test_library_distribution = Counter(get_label_distribution(test_data, 'visual_node_types'))

    # Generate bar chart for library functions distribution
    all_visual_node_types = sorted(set(
        list(train_library_distribution.keys()) +
        list(validation_library_distribution.keys()) +
        list(test_library_distribution.keys())
    ))
    x_lib = np.arange(len(all_visual_node_types))  # the label locations
    width = 0.2  # the width of the bars

    train_library_counts = [train_library_distribution.get(func, 0) for func in all_visual_node_types]
    validation_library_counts = [validation_library_distribution.get(func, 0) for func in all_visual_node_types]
    test_library_counts = [test_library_distribution.get(func, 0) for func in all_visual_node_types]

    # Plotting code
    plt.figure(figsize=(10, 6))
    plt.bar(x_lib - width, train_library_counts, width, label='Train')
    plt.bar(x_lib, validation_library_counts, width, label='Validation')
    plt.bar(x_lib + width, test_library_counts, width, label='Test')
    plt.ylabel('Count')
    plt.title('Visual Node types Distribution by Dataset')
    plt.xticks(x_lib, all_visual_node_types, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_functions_distribution(train_data, validation_data, test_data):
    train_library_distribution = Counter(get_label_distribution(train_data, 'external_functions'))
    validation_library_distribution = Counter(get_label_distribution(validation_data, 'external_functions'))
    test_library_distribution = Counter(get_label_distribution(test_data, 'external_functions'))

    # Generate bar chart for library functions distribution
    all_external_functions = sorted(set(
        list(train_library_distribution.keys()) +
        list(validation_library_distribution.keys()) +
        list(test_library_distribution.keys())
    ))
    x_lib = np.arange(len(all_external_functions))  # the label locations
    width = 0.2  # the width of the bars

    train_library_counts = [train_library_distribution.get(func, 0) for func in all_external_functions]
    validation_library_counts = [validation_library_distribution.get(func, 0) for func in all_external_functions]
    test_library_counts = [test_library_distribution.get(func, 0) for func in all_external_functions]

    # Plotting code
    plt.figure(figsize=(10, 6))
    plt.bar(x_lib - width, train_library_counts, width, label='Train')
    plt.bar(x_lib, validation_library_counts, width, label='Validation')
    plt.bar(x_lib + width, test_library_counts, width, label='Test')
    plt.ylabel('Count')
    plt.title('External Functions Distribution by Dataset')
    plt.xticks(x_lib, all_external_functions, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_instance_distribution(train_data, validation_data, test_data):
    train_instance_distribution = Counter(get_label_distribution(train_data, 'textual_instance_types'))
    validation_instance_distribution = Counter(get_label_distribution(validation_data, 'textual_instance_types'))
    test_instance_distribution = Counter(get_label_distribution(test_data, 'textual_instance_types'))

    # Generate bar chart for textual instance types distribution
    all_instance_types = sorted(set(
        list(train_instance_distribution.keys()) +
        list(validation_instance_distribution.keys()) +
        list(test_instance_distribution.keys())
    ))
    x_inst = np.arange(len(all_instance_types))  # the label locations
    width = 0.2  # the width of the bars

    train_instance_counts = [train_instance_distribution.get(inst, 0) for inst in all_instance_types]
    validation_instance_counts = [validation_instance_distribution.get(inst, 0) for inst in all_instance_types]
    test_instance_counts = [test_instance_distribution.get(inst, 0) for inst in all_instance_types]

    # Plotting code
    plt.figure(figsize=(10, 6))
    plt.bar(x_inst - width, train_instance_counts, width, label='Train')
    plt.bar(x_inst, validation_instance_counts, width, label='Validation')
    plt.bar(x_inst + width, test_instance_counts, width, label='Test')
    plt.ylabel('Count')
    plt.title('Instance Types Distribution by Dataset')
    plt.xticks(x_inst, all_instance_types, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
# if __name__ == '__main__':
    # Load your datasets
    # train_data = ...
    # validation_data = ...
    # test_data = ...

    # analyze_library_distribution(train_data, validation_data, test_data)
    # analyze_instance_distribution(train_data, validation_data, test_data)