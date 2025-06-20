import json
from collections import Counter
from utils import parse_args

def create_keyset(file_path, output_path):
    with open(file_path) as f:
        lines = f.readlines()

    users = [f"User {int(line.split()[0])-1}" for line in lines]
    keyset = {
        "item_num": max(int(i) for line in lines for i in line.split()[1:]),
        "train": users,
        "val": users,
        "test": users
    }

    with open(output_path, 'w') as f:
        json.dump(keyset, f, indent=2)

def create_future(input_file, output_file):
    data = {}
    with open(input_file) as f:
        for line in f:
            tokens = line.strip().split()
            uid = f"User {int(tokens[0])-1}"
            items = list(map(int, tokens[1:]))
            data[uid] = [[-1], [items[-1]], [-1]]
    with open(output_file, 'w') as f:
        json.dump(data, f)

def create_rel(input_file, output_file):
    user_relevancy = {}
    with open(input_file) as f:
        for line in f:
            tokens = line.strip().split()
            uid = f"User {int(tokens[0])-1}"
            items = list(map(int, tokens[1:]))
            user_relevancy[uid] = {items[-1]: 1}
    with open(output_file, 'w') as f:
        json.dump(user_relevancy, f)


def create_group_purchase_file(input_file, output_file, top_percent=0.2):
    item_counter = Counter()

    # Count all item frequencies
    with open(input_file) as f:
        for line in f:
            items = line.strip().split()[1:]  # skip user ID
            item_counter.update(items)

    # Sort by frequency
    sorted_items = [item for item, _ in item_counter.most_common()]
    total_items = len(sorted_items)
    top_k = int(total_items * top_percent)

    group_item = {
        "pop": sorted_items[:top_k],
        "unpop": sorted_items[top_k:]
    }

    with open(output_file, 'w') as f:
        json.dump(group_item, f, indent=2)

    print(f"Saved to {output_file}. Pop: {len(group_item['pop'])}, Unpop: {len(group_item['unpop'])}")



def main(args):
    # args = parse_args() # dataset and filepath

    create_keyset(f"data/{args.data_name}.txt", f"data/{args.data_name}_keyset.json")
    create_future(f"data/{args.data_name}.txt", f"data/{args.data_name}_future.json")
    create_rel(f"data/{args.data_name}.txt", f"data/{args.data_name}_rel.json")
    create_group_purchase_file(f"data/{args.data_name}.txt", f"data/{args.data_name}_group_purchase.json", top_percent=0.2)

if __name__ == "__main__":
    args = parse_args()
    print("Starting generation...")
    main(args)