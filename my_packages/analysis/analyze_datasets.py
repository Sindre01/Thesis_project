import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_datasets_pca(datasets, dataset_labels=None, title="Dataset Embeddings Visualization (PCA)"):
    """
    Visualize multiple embedding datasets in the same 2D PCA plot with different colors.

    Args:
        datasets (List[np.ndarray]): List of arrays (each of shape N_i x D), where each array is one dataset's embeddings.
        dataset_labels (List[str], optional): List of labels corresponding to each dataset. If None, generic labels will be used.
        title (str): Title of the plot.
    """
    if dataset_labels is None:
        dataset_labels = [f"Dataset {i+1}" for i in range(len(datasets))]

    # Concatenate all embeddings
    try:
        all_embeddings = np.vstack(datasets)
    except Exception as e:
        print("❌ Error while stacking embeddings. Ensure all inputs are 2D arrays.")
        print(e)
        return

    # Check for minimum dimension
    if all_embeddings.shape[1] < 2:
        print("❌ Not enough dimensions in embeddings for PCA (need at least 2).")
        return

    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_embeddings)

    # Plot each dataset in different color
    plt.figure(figsize=(10, 6))
    start = 0
    for i, data in enumerate(datasets):
        n = len(data)
        plt.scatter(
            projected[start:start + n, 0],
            projected[start:start + n, 1],
            label=dataset_labels[i],
            alpha=0.7
        )
        start += n
    print("Visualizing PCA projection of embeddings")
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout

def get_label_distribution(data_subset, label_type):
    subset_labels = []
    for item in data_subset:
        labels = item.get(label_type, [])
        #remove "root.std" from the labels string
        labels = [label.replace("root.std.", "") for label in labels]
        subset_labels.extend(labels)
    return subset_labels

def analyze_visual_node_types_distribution(train_data, validation_data, test_data):
    train_dist = Counter(get_label_distribution(train_data, 'visual_node_types'))
    test_dist  = Counter(get_label_distribution(test_data,  'visual_node_types'))

    # Only compute validation_dist if validation_data is not None
    if validation_data is not None:
        validation_dist = Counter(get_label_distribution(validation_data, 'visual_node_types'))
    else:
        validation_dist = None

    # Gather all possible keys
    all_keys = set(train_dist.keys()) | set(test_dist.keys())
    if validation_dist is not None:
        all_keys |= set(validation_dist.keys())

    all_keys = sorted(all_keys)
    x_pos = np.arange(len(all_keys))  # the label locations
    width = 0.2

    # Prepare counts
    train_counts = [train_dist.get(k, 0) for k in all_keys]
    test_counts  = [test_dist.get(k, 0)  for k in all_keys]
    if validation_dist is not None:
        validation_counts = [validation_dist.get(k, 0) for k in all_keys]

    # Plot
    plt.figure(figsize=(10, 6))
    # Train always shown
    plt.bar(x_pos - width, train_counts, width, label='Train')

    # If validation_data is not None, plot it and shift test accordingly
    if validation_dist is not None:
        plt.bar(x_pos, validation_counts, width, label='Validation')
        plt.bar(x_pos + width, test_counts, width, label='Test')
    else:
        # If no validation_data, only plot test next to train
        plt.bar(x_pos, test_counts, width, label='Test')

    plt.ylabel('Count')
    plt.title('Visual Node Types Distribution by Dataset')
    plt.xticks(x_pos, all_keys, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_functions_distribution(train_data, validation_data, test_data):
    train_dist = Counter(get_label_distribution(train_data, 'external_functions'))
    test_dist  = Counter(get_label_distribution(test_data,  'external_functions'))

    # Only compute validation_dist if validation_data is not None
    if validation_data is not None:
        validation_dist = Counter(get_label_distribution(validation_data, 'external_functions'))
    else:
        validation_dist = None

    # Gather all possible keys
    all_keys = set(train_dist.keys()) | set(test_dist.keys())
    if validation_dist is not None:
        all_keys |= set(validation_dist.keys())

    all_keys = sorted(all_keys)
    x_pos = np.arange(len(all_keys))
    width = 0.2

    # Prepare counts
    train_counts = [train_dist.get(k, 0) for k in all_keys]
    test_counts  = [test_dist.get(k, 0)  for k in all_keys]
    if validation_dist is not None:
        validation_counts = [validation_dist.get(k, 0) for k in all_keys]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(x_pos - width, train_counts, width, label='Train')
    if validation_dist is not None:
        plt.bar(x_pos, validation_counts, width, label='Validation')
        plt.bar(x_pos + width, test_counts, width, label='Test')
    else:
        plt.bar(x_pos, test_counts, width, label='Test')

    plt.ylabel('Count')
    plt.title('External Functions Distribution by Dataset')
    plt.xticks(x_pos, all_keys, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_instance_distribution(train_data, validation_data, test_data):
    train_dist = Counter(get_label_distribution(train_data, 'textual_instance_types'))
    test_dist  = Counter(get_label_distribution(test_data,  'textual_instance_types'))

    # Only compute validation_dist if validation_data is not None
    if validation_data is not None:
        validation_dist = Counter(get_label_distribution(validation_data, 'textual_instance_types'))
    else:
        validation_dist = None

    # Gather all possible keys
    all_keys = set(train_dist.keys()) | set(test_dist.keys())
    if validation_dist is not None:
        all_keys |= set(validation_dist.keys())

    all_keys = sorted(all_keys)
    x_pos = np.arange(len(all_keys))
    width = 0.2

    # Prepare counts
    train_counts = [train_dist.get(k, 0) for k in all_keys]
    test_counts  = [test_dist.get(k, 0)  for k in all_keys]
    if validation_dist is not None:
        validation_counts = [validation_dist.get(k, 0) for k in all_keys]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(x_pos - width, train_counts, width, label='Train')
    if validation_dist is not None:
        plt.bar(x_pos, validation_counts, width, label='Validation')
        plt.bar(x_pos + width, test_counts, width, label='Test')
    else:
        plt.bar(x_pos, test_counts, width, label='Test')

    plt.ylabel('Count')
    plt.title('Instance Types Distribution by Dataset')
    plt.xticks(x_pos, all_keys, rotation=90)
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