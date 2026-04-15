import pandas as pd
import csv

def read_csv_to_list(file_path):
    try:

        df = pd.read_csv(file_path, header=None)

        float_list = df[0].tolist()
        return float_list
    except FileNotFoundError:
        print(f"error: {file_path}")
        return []
    except Exception as e:
        print(e)
        return []

def merged_csv_to_list(file_paths_tuple):

    merged_data = []
    for file_path in file_paths_tuple:
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        merged_data.append(row[0])
        except FileNotFoundError:
            print(f"error：{file_path}")
        except Exception as e:
            print(e)
    return merged_data