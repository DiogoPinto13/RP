# import os
# import pandas as pd

# def find_best_feature_selection(csv_file):
#     try:
#         df = pd.read_csv(csv_file)
#         if "numberFeaturesSelection" in df.columns and "fScore" in df.columns:
#             best_row = df.loc[df["fScore"].idxmax()]
#             return best_row["numberFeaturesSelection"], best_row["fScore"]
#         else:
#             print(f"Skipping {csv_file}: Required columns not found.")
#             return None, None
#     except Exception as e:
#         print(f"Error reading {csv_file}: {e}")
#         return None, None

# def process_csv_files(directory):
#     results = {}
#     for file in os.listdir(directory):
#         if file.endswith(".csv"):
#             file_path = os.path.join(directory, file)
#             best_feature_selection, best_f1 = find_best_feature_selection(file_path)
#             if best_feature_selection is not None:
#                 results[file] = (best_feature_selection, best_f1)
#     return results

# if __name__ == "__main__":
#     #directory = input("Enter the directory path containing CSV files: ")
#     directory = "./outputs/cfd_bkp2/"
#     results = process_csv_files(directory)
    
#     for file, (best_feature_selection, best_f1) in results.items():
#         print(f"File: {file} -> Best numberFeaturesSelection: {best_feature_selection} with f1Score: {best_f1}")

import os
import pandas as pd

def find_best_feature_selection(csv_file):
    try:
        df = pd.read_csv(csv_file)
        if "numberFeaturesSelection" in df.columns and "fScore" in df.columns:
            best_row = df.loc[df["fScore"].idxmax()]
            return best_row["numberFeaturesSelection"], best_row["fScore"], df["fScore"].mean()
        else:
            print(f"Skipping {csv_file}: Required columns not found.")
            return None, None, None
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None, None, None

def process_csv_files(directory):
    results = {}
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            best_feature_selection, best_f1, avg_f1 = find_best_feature_selection(file_path)
            if best_feature_selection is not None:
                results[file] = (best_feature_selection, best_f1, avg_f1)
    return results

if __name__ == "__main__":
    directory = directory = "./outputs/comparisonClassifiers_bkp3/"
    results = process_csv_files(directory)
    
    for file, (best_feature_selection, best_f1, avg_f1) in results.items():
        print(f"{file} -> Best numberFeaturesSelection: {best_feature_selection} with f1Score: {best_f1}, Avg f1Score: {avg_f1}")
