import gzip
import json
from pathlib import Path
from collections import Counter
import numpy as np
print(np.__version__)

review_path = Path("../data/raw/reviews_Electronics.jsonl.gz")

review_count = 0
users = set()
items = set()
rating_counts = Counter()

user_counts = Counter()
item_counts = Counter()


with gzip.open(review_path, "rt", encoding="utf-8") as f:
    for line in f:
        review = json.loads(line)

        review_count += 1

        user_id = review["user_id"]
        item_id = review["parent_asin"]
        rating = review["rating"]

        users.add(user_id)
        items.add(item_id)

        rating_counts[rating] += 1

        user_counts[user_id] += 1
        item_counts[item_id] += 1


print("Review count:", review_count)
print("Unique users:", len(users))
print("Unique items:", len(items))

print("\nRating distribution:")
for rating in sorted(rating_counts):
    print(rating, ":", rating_counts[rating])


print("\nUser interaction statistics:")
print("Min:", min(user_counts.values()))
print("Max:", max(user_counts.values()))
print("Average:", sum(user_counts.values()) / len(user_counts))


print("\nItem interaction statistics:")
print("Min:", min(item_counts.values()))
print("Max:", max(item_counts.values()))
print("Average:", sum(item_counts.values()) / len(item_counts))


# Percentiles
user_values = list(user_counts.values())
item_values = list(item_counts.values())

print("\nUser interaction percentiles:")
print("P50:", np.percentile(user_values, 50))
print("P90:", np.percentile(user_values, 90))
print("P95:", np.percentile(user_values, 95))
print("P99:", np.percentile(user_values, 99))

print("\nItem interaction percentiles:")
print("P50:", np.percentile(item_values, 50))
print("P90:", np.percentile(item_values, 90))
print("P95:", np.percentile(item_values, 95))
print("P99:", np.percentile(item_values, 99))

# 统计不同交互频次下，分别有多少用户、多少商品。
def count_frequency_distribution(counts):
    distribution = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6-10": 0,
        "11-20": 0,
        "21-50": 0,
        "51-100": 0,
        ">100": 0,
    }

    for count in counts.values():
        if count == 1:
            distribution["1"] += 1
        elif count == 2:
            distribution["2"] += 1
        elif count == 3:
            distribution["3"] += 1
        elif count == 4:
            distribution["4"] += 1
        elif count == 5:
            distribution["5"] += 1
        elif count <= 10:
            distribution["6-10"] += 1
        elif count <= 20:
            distribution["11-20"] += 1
        elif count <= 50:
            distribution["21-50"] += 1
        elif count <= 100:
            distribution["51-100"] += 1
        else:
            distribution[">100"] += 1

    return distribution


user_distribution = count_frequency_distribution(user_counts)
item_distribution = count_frequency_distribution(item_counts)

# 有多少用户属于每个交互次数区间
print("\nUser interaction frequency distribution:")
for group, count in user_distribution.items():
    print(f"{group}: {count}")

# 有多少商品属于每个被交互次数区间
print("\nItem interaction frequency distribution:")
for group, count in item_distribution.items():
    print(f"{group}: {count}")