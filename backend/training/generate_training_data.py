import pandas as pd

from services.feature_extractor import feature_extractor
from services.bm25_service import bm25_search
from services.retrieval_service import dense_search

training_data = []

queries = [

    "Explain Agile Scrum",

    "Scrum roles",

    "Sprint planning",

    "Agile methodology",

    "Product backlog",

    "Daily scrum",

    "Sprint review",

    "Scrum master responsibilities",

    "What is sprint backlog?",

    "Difference between Agile and Scrum"

]

for query in queries:

    features = feature_extractor.extract(query)

    bm25_docs, bm25_scores = bm25_search(query)

    dense_docs, dense_scores = dense_search(query)

    bm25_score = sum(bm25_scores) / len(bm25_scores)

    dense_score = sum(dense_scores) / len(dense_scores)

    total = bm25_score + dense_score

    if total == 0:
        continue

    bm25_weight = bm25_score / total

    dense_weight = dense_score / total

    row = features.copy()

    row["bm25_weight"] = bm25_weight

    row["dense_weight"] = dense_weight

    training_data.append(row)

df = pd.DataFrame(training_data)

df.to_csv("training_dataset.csv", index=False)

print(df.head())