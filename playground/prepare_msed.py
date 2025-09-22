import csv
import json

csv_file = 'playground/data/MSED/train/train.csv'
json_file = 'playground/data/MSED/train.json'

data = []
with open(csv_file, mode='r', encoding='utf-8') as f:
    csv_reader = csv.DictReader(f)
    for idx, row in enumerate(csv_reader, start=1):
        item = {
            "idx": idx,
            "title": row["Title"],
            "caption": row["Caption"],
            "sentiment": row["Sentiment"],
            "emotion": row["Emotion"],
            "desire": row["Desire"],
            "inference": row["Inference Sequence"],
            "image": f"{idx}.jpg"
        }
        data.append(item)

# 写入JSON文件
with open(json_file, mode='w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"trans finish, saving to {json_file}")
