import csv
import json

csv_file = 'prediction/icl_gpt-4-1106-preview_onestop_char.csv'
jsonl_file = 'results-icl/icl_gpt-4-1106-preview_onestop_char.jsonl'


def csv_to_jsonl(csv_file, jsonl_file):
    with open(csv_file, mode='r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        with open(jsonl_file, mode='w') as jsonl_file:
            for row in csv_reader:
                json_line = json.dumps(row)
                jsonl_file.write(json_line + '\n')


if __name__ == "__main__":
    csv_to_jsonl(csv_file, jsonl_file)
