import h5py
import numpy as np
import os
import pandas as pd


def print_h5_contents(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        def print_attrs(name, obj):
            print(f"Name: {name}")
            for key, val in obj.attrs.items():
                print(f"  Attribute - {key}: {val}")
            if isinstance(obj, h5py.Dataset):
                print(f"  Data: {obj[:]}\n")
            elif isinstance(obj, h5py.Group):
                print("  Group contains:")
                for key in obj.keys():
                    print(f"    {key}")
                print()
                
        h5_file.visititems(print_attrs)


def merge_files(directory):
    all_npy = []
    all_csv = []
    mine_npy = []
    mine_csv = []
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        
        if file.endswith(".npy"):
            data = np.load(file_path)
            if "mine" in file:
                mine_npy.append(data)
            else:
                all_npy.append(data)
        elif file.endswith(".csv"):
            data = pd.read_csv(file_path)
            if "mine" in file:
                mine_csv.append(data)
            else:
                all_csv.append(data)
    
    if mine_npy or all_npy:
        merged_npy = np.vstack(mine_npy + all_npy)
        np.save(os.path.join(directory, "all_embeddings.npy"), merged_npy)
    
    if mine_csv or all_csv:
        merged_csv = pd.concat(mine_csv + all_csv)
        merged_csv.to_csv(os.path.join(directory, "all_metadata.csv"), index=False)




merge_files('/Users/cemerturkan/Desktop/personal_projects/data/output_embed/')