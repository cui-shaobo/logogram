import jsonlines

def concat_nips_datasets():
    final_results = []
    for year in range(1987, 2022):
        with jsonlines.open(f'../results/data_nips_{year}.jsonl', 'r') as reader:
            for sample in reader.iter():
                final_results.append(sample)
        print(len(final_results))

    with jsonlines.open(f'../results/data_nips.jsonl', 'w') as f:
        for sample in final_results:
            f.write(sample)


def concat_acl_datasets():
    final_results = []
    for year in range(2016, 2023):
        with jsonlines.open(f'./results/data_acl_{year}.jsonl', 'r') as reader:
            for sample in reader.iter():
                final_results.append(sample)
        print(len(final_results))

    with jsonlines.open(f'./results/data_acl.jsonl', 'w') as f:
        for sample in final_results:
            f.write(sample)

if __name__ == '__main__':
    concat_acl_datasets()