

import os
import sys
sys.path.append('../..')
from my_packages.data_processing.attributes_processing import used_functions_from_dataset
from my_packages.data_processing.split_dataset import create_kfold_splits
from my_packages.prompting.few_shot import transform_code_data
from my_packages.utils.file_utils import read_dataset_to_json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../..")
main_dataset_folder = f'{project_dir}/data/MBPP_Midio_50/'

print("\n==== Splits data ====")

def get_hold_out_splits(main_dataset_folder):
    train_data = transform_code_data(read_dataset_to_json(main_dataset_folder + 'splits/hold_out/train_dataset.json'))
    val_data = transform_code_data(read_dataset_to_json(main_dataset_folder + 'splits/hold_out/validation_dataset.json'))
    test_data = transform_code_data(read_dataset_to_json(main_dataset_folder + 'splits/hold_out/test_dataset.json'))

    print(f"Train data: {len(train_data)}")
    print(f"Val data: {len(val_data)}")
    print(f"Test data: {len(test_data)}")
    return train_data, val_data, test_data

# def get_k_fold_splits(main_dataset_folder, k, k_folds=5):

#     if os.path.exists(main_dataset_folder + f'splits/{k_folds}_fold'):
#         print(f"Using existing {k_folds}-fold splits")
#     else:
#         print(f"Creating {k_folds}-fold splits")
#         train, _ , test = get_hold_out_splits(main_dataset_folder)
#         create_kfold_splits((test+train), k_folds=k_folds, write_to_file=True)

#     train_data = read_dataset_to_json(main_dataset_folder + f'splits/{k_folds}_fold/train_dataset_{k}.json')
#     test_data = read_dataset_to_json(main_dataset_folder + f'splits/{k_folds}_fold/test_dataset_{k}.json')

#     print(f"Train data: {len(train_data)}")
#     print(f"Test data: {len(test_data)}")
#     return train_data, test_data

# get_k_fold_splits(main_dataset_folder, 1, k_folds=5)
# train, _ , test = get_hold_out_splits(main_dataset_folder)
# data, folds = create_kfold_splits((train+test), k_folds=3, write_to_file=True)
# print(folds)
# main_dataset_folder = f'{project_dir}/data/MBPP_Midio_50/'
dataset = read_dataset_to_json(main_dataset_folder +"MBPP-Midio-50.json")
# j = []
used_functions_from_dataset(dataset, write_to_file=True)
# for fold in range(3):
#     train_data = [data[i] for i in folds[fold][0]]
#     test_data = [data[i] for i in folds[fold][1]]

#     print(f"Fold {fold+1}")
#     print(f"len(train_data): {len(train_data)}")
#     for item in train_data:
#         print(item['prompts'][0])
    

#     embedder = OllamaEmbeddings(model="nomic-embed-text")
#     texts = [example["prompts"][0] for example in train_data[:4]]
#     vectors = embedder.embed_documents(texts)
#     j.append(vectors)

#     # pca = PCA(n_components=2)
#     # projected = pca.fit_transform(vectors)

#     # plt.figure(figsize=(8,6))
#     # plt.scatter(projected[:, 0], projected[:, 1])
#     # plt.title("PCA Projection of Prompt Embeddings")
#     # plt.xlabel("PC 1")
#     # plt.ylabel("PC 2")
#     # plt.show()
#     # print(f"len(test_data): {len(test_data)}")
#     # analyze_functions_distribution(train, [], test)
#     # analyze_instance_distribution(train, [], test)
#     # analyze_visual_node_types_distribution(train, [], test) 
#     print(len(j))
#     visualize_datasets_pca(j, ["Fold 1", "Fold 2", "Fold 3"], "PCA Projection of Prompt Embeddings")