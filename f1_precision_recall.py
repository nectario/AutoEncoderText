import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from common.DataUtils import *
from Evaluate import *

if __name__ == "__main__":

    id_to_genre = {}
    genre_to_id = {}

    genres = load_genre_labels(file_path="data/new_48_genres.xlsx")

    for i, genre in enumerate(genres):
        genre_to_id[genre] = i
        id_to_genre[i] = genre

    labels = pd.read_csv("data/test_20k_48.csv").groupby("Id").mean()
    predictions = pd.read_excel("data/pluto_vectors/20_validation_vectors_20.xlsx").set_index("Id")
    for genre in genres:
        predictions[genre] = predictions[genre].apply(convert_binary)

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

    predictions["Predictions"] = create_comma_separated_str_values(predictions, genres)
    predictions["Labels"] = create_comma_separated_str_values(labels, genres)

    print("Macro F1:",macro_f1_score)
    print("Micro F1:", micro_f1_score)
    print()
    print("Macro Recall:", macro_recall_score)
    print("Micro Recall:", micro_recall_score)
    print()
    print("Macro Precision:", macro_precision_score)
    print("Micro Precision:", micro_precision_score)
    print()

    evaluate(predictions)

    metrics_df.to_excel("data/precision_recall_f1_multi_label.xlsx", index=False)
    print("Done")