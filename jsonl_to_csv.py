import json
import pandas as pd

def jsonl_to_dataframe(jsonl_file_path):
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        jsonl_content = file.readlines()
    data = [json.loads(line) for line in jsonl_content]
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    filename = 'results-icl/icl_gpt-4-1106-preview_onestop.'
    jsonl_file = filename + 'jsonl'
    csv_file_path = filename + 'csv'

    df = jsonl_to_dataframe(jsonl_file)

    df.to_csv(csv_file_path, index=False)

