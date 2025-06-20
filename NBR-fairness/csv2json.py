import pandas as pd
import json

def convert_csv_to_json(csv_path, json_path):
    df = pd.read_csv(csv_path)

    result = {
        f"User {row['user_id']}" : [int(x.strip()) for x in row['item_id_predictions'].split(',')]
        for _, row in df.iterrows()
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

# Example usage
model = 'SASRec'
csv_file = f'predictions_{model}/LastFM_pred.csv'
json_file = f'predictions_{model}/LastFM_pred.json'
convert_csv_to_json(csv_file, json_file)
