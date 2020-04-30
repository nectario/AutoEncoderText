import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from common.data_utils import *

if __name__ == "__main__":

    id_to_genre = {}
    genre_to_id = {}

    genres = load_genres(file_path="data/93_labels_data/new_93_genres.xlsx")

    for i, genre in enumerate(genres):
        genre_to_id[genre] = i
        id_to_genre[i] = genre

    labels = pd.read_csv("data/test_20k_48_Labels.csv").groupby("Id").mean()
    predictions = pd.read_excel("data/binary_validation_vectors_20.xlsx").set_index("Id")


    metrics_df = pd.DataFrame(columns=["Label", "Precision", "Recall", "F1"])

    metrics_df["Label"] = genres

    precision_scores = precision_score(labels,predictions, average=None)
    metrics_df["Precision"] = precision_scores

    recall_scores = recall_score(labels,predictions, average=None)
    metrics_df["Recall"] = recall_scores

    f1_scores =  f1_score(labels,predictions, average=None)
    metrics_df["F1"] = f1_scores

    macro_f1_score = f1_score(labels,predictions, average="macro")
    micro_f1_score = f1_score(labels,predictions, average="micro")

    macro_recall_score = recall_score(labels,predictions, average="macro")
    micro_recall_score = recall_score(labels,predictions, average="micro")

    macro_precision_score = precision_score(labels,predictions, average="macro")
    micro_precision_score = precision_score(labels,predictions, average="micro")

    metrics_df.to_excel("data/precision_recall_f1.xlsx", index=False)
    print("Done")