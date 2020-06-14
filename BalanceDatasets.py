
import pandas as pd
from common.DataUtils import *
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
from skmultilearn.dataset import load_dataset
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split

ids_to_binary_labels ={}

def remove_unused_genres(genre_list, genres):
    if (type(genre_list) == str):
        genre_list = parse_str_labels(genre_list)
        new_list = list(genre_list)
        for genre in genre_list:
            if "comedy" in genre:
                if "-" in genre:
                    split_genre = genre.split("-")
                    part1 = split_genre[0]
                    part2 = split_genre[1]
                    if part1 in genres and part2 in genres:
                        new_list.extend([part1,part2])
                        new_list.remove(genre)
                else:
                    if " " in genre:
                        split_genre = genre.split(" ")
                        if len(split_genre) == 2:
                            part1 = split_genre[0]
                            part2 = split_genre[1]
                            if part1 in genres and part2 in genres:
                                new_list.extend([part1, part2])
                                new_list.remove(genre)

        genres_set = OrderedSet(list(genres))
        genre_list_set = OrderedSet(list(new_list))
        output = list(genre_list_set.intersection(genres_set))
        return str(output).replace("[","").replace("]","").replace("'","")
    else:
        return ""

def get_all_genres_from_data(data_df, genre_column=None, parse=True):
    """
    Get the list of all genres without removing the duplicates. This will be used to count the
    frequency of each genre.
    :param data_df:
    :param genre_colum:
    :param parse:
    :return: all genres dataframe
    """
    all_genres = []
    for i, row in data_df.iterrows():
        if parse:
            all_genres.extend(split_string_list(row[genre_column]))
        else:
            all_genres.append(row[genre_column])
    return all_genres

def count_genres(data_df=None, return_dataframe=False, genre_column="Label", columns=["Genre", "Count"]):
    """
    Perform a frequency count on all genres.
    :return:
    """
    genre_list = get_all_genres_from_data(data_df, genre_column=genre_column)

    genre_count_map = Counter(genre_list)
    genre_count_df = pd.DataFrame(list(genre_count_map.items()), columns=columns).sort_values(by=["Count"], ascending=False)
    # genre_count_df.replace("romantic ", "romance ", regex=True, inplace=True)

    if return_dataframe:
        return genre_count_map, genre_count_df,
    else:
        return genre_count_map


def random_min_split(all_data_df):

    all_data_df = all_data_df.reset_index()
    labels = create_comma_separated_str_values(all_data_df, genres)
    all_data_df["Labels"] = labels
    all_data_df.set_index("Id")
    all_data_df["Id"] = all_data_df.index
    # all_data_df.to_excel("data/all_data.xlsx", index=False)
    labels = all_data_df[genres].values
    # counter = Counter(combination for row in get_combination_wise_output_matrix(labels, order=2) for combination in row)
    # for i, v in counter.items():
    #    if i[0] != i[1]:
    #        print(genres[i[0]],genres[i[1]], v)
    X = all_data_df.values
    y = labels
    # iterative_train_test_split(X, y, test_size=0.20)
    num_splits = 300
    split_list = []
    train_std_deviations = []
    test_std_deviations = []
    diff_min_max_train = []
    diff_min_max_test = []
    for i in range(num_splits):
        X_train, X_test, y_train, y_test = train_test_split(all_data_df, all_data_df["Labels"], test_size=0.20, shuffle=True)
        split_list.append((X_train, y_train, X_test, y_test))

        train_count_map = count_genres(X_train)
        test_count_map = count_genres(X_test)

        train_val = list(train_count_map.values())
        test_val = list(test_count_map.values())

        train_standard_deviation = np.std(train_val)
        test_standard_deviation = np.std(test_val)

        diff_min_max_train.append(min(train_val))
        diff_min_max_test.append(min(test_val))
        train_std_deviations.append(train_standard_deviation)
        test_std_deviations.append(test_standard_deviation)

        print("Split:", i)
        print("Train:", count_genres(X_train))
        print("Test:", count_genres(X_test))
        print("--------------------------------------------------------")
    index_train = np.argmax(diff_min_max_train).squeeze()
    index_test = np.argmax(diff_min_max_test).squeeze()
    X_train, y_train, X_test, y_test = split_list[index_train]
    X_train_t, y_train_t, X_test_t, y_test_t = split_list[index_test]
    X_train.to_excel("data/train_new.xlsx", index=False)
    X_test.to_excel("data/validation_new.xlsx", index=False)
    X_train_t.to_excel("data/train_new_t.xlsx", index=False)
    X_test_t.to_excel("data/validation_new_t.xlsx", index=False)
    print("Final Selection Split:", index_train)
    genre_count_map, genre_count_df_train = count_genres(X_train, return_dataframe=True)
    genre_count_map_test, genre_count_df_test = count_genres(X_test, return_dataframe=True)
    genre_count_df_train.to_excel("data/genre_count_train_strat_train.xlsx", index=False)
    genre_count_df_test.to_excel("data/genre_count_test_strat_train.xlsx", index=False)
    print(count_genres(X_train))
    print(count_genres(X_test))
    print("-------")
    print("Final Selection Split Test:", index_test)
    genre_count_map_t, genre_count_df_train_t = count_genres(X_train_t, return_dataframe=True)
    genre_count_map_test_t, genre_count_df_test_t = count_genres(X_test_t, return_dataframe=True)
    genre_count_df_train_t.to_excel("data/genre_count_train_strat_test_2.xlsx", index=False)
    genre_count_df_test_t.to_excel("data/genre_count_test_strat_test_2.xlsx", index=False)
    print(count_genres(X_train_t))
    print(count_genres(X_test_t))
    return X_train, X_test



def create_new_gold_data(genres, output_file):
    all_data_df = pd.read_excel("data/film_data_lots.xlsx")
    all_data_df = all_data_df.set_index("Id")
    all_data_df["Id"] = all_data_df.index

    #train_data_df = pd.read_excel("data/train_data.xlsx")
    #validation_data_df = pd.read_excel("data/validation_data.xlsx")
    #new_data_df = pd.read_excel("data/new_data.xlsx")

    #all_data_df = pd.DataFrame(columns=train_data_df.columns)
    #all_data_df = all_data_df.append(train_data_df)
    #all_data_df = all_data_df.append(validation_data_df)
    #all_data_df = all_data_df.append(new_data_df)

    all_data_df["Labels"] = all_data_df["Genres"].apply(remove_unused_genres, args=(genres,)).tolist()

    all_data_df = all_data_df.drop(columns=["Genres"], axis=1)

    all_data_df["List Labels"] = all_data_df["Labels"].apply(parse_str_labels)

    all_data_df =  all_data_df[all_data_df.Labels != ""]

    count_map, count_df = count_genres(all_data_df, return_dataframe=True)

    all_data_df = all_data_df.set_index("Id")
    all_data_df["Id"] = all_data_df.index
    all_data_df["Title"] = all_data_df["Title"]

    count_df.to_excel("data/new_100_genres.xlsx", index=False)
    all_data_df.to_excel("data/debug_all_data.xlsx")

    all_data_df = generate_binary_columns(all_data_df, genres=genres, label_column="List Labels")

    new_columns = ["Id", "Title", "Labels", "IsTop5K", "Subtitles 1", "Subtitles 2"]
    new_columns.extend(genres)
    all_data_df = all_data_df[new_columns]

    all_data_df.to_excel(output_file, index=False)
    return all_data_df


def stratify_split_2(data_df, label_column ="Label"):
    data_df = data_df.set_index("Id")
    data_df["Id"] = data_df.index

    all_data_count = data_df.groupby(label_column).count()
    all_data_count = all_data_count.sort_values(by="Id", ascending=False)
    all_data_groups = data_df.groupby(label_column)
    validation_data_df = pd.DataFrame(columns=data_df.columns)

    for i, row in all_data_count.iterrows():
        group = all_data_groups.get_group(i)
        group_no_5K = group[group.IsTop5K != True]

        val_data = group_no_5K.sample(frac=0.18)
        validation_data_df = validation_data_df.append(val_data)

        print("Group:", i)

    validation_data_df = validation_data_df.sort_values(by="Id")
    validation_data_df = validation_data_df.set_index("Id")
    validation_data_df["Id"] = validation_data_df.index

    train_data_df = data_df[~data_df.index.isin(validation_data_df.index)]

    return train_data_df, validation_data_df


def stratify_split(new_gold_data_df, label_column = "Label"):
    new_gold_data_df = new_gold_data_df.set_index("Id")
    new_gold_data_df["Id"] = new_gold_data_df.index

    all_data_count = new_gold_data_df.groupby(label_column).count()
    all_data_count = all_data_count.sort_values(by="Id", ascending=False)
    all_data_groups = new_gold_data_df.groupby(label_column)
    validation_data_df = pd.DataFrame(columns=new_gold_data_df.columns)

    for i, row in all_data_count.iterrows():
        group = all_data_groups.get_group(i)
        group_no_5K = group[group.IsTop5K != True]
        if group_no_5K.shape[0] >= 120:
            val_data = group_no_5K.sample(frac=0.2)
            validation_data_df = validation_data_df.append(val_data)
        elif group_no_5K.shape[0] < 120 and group_no_5K.shape[0] >= 40:
            val_data = group_no_5K.sample(frac=0.17)
            validation_data_df = validation_data_df.append(val_data)
        print("Group:", i)

    validation_data_df = validation_data_df.sort_values(by="Id")
    validation_data_df = validation_data_df.set_index("Id")
    validation_data_df["Id"] = validation_data_df.index

    train_data_df = new_gold_data_df[~new_gold_data_df.index.isin(validation_data_df.index)]
    validation_genre_count, val_count_df = count_genres(validation_data_df, return_dataframe=True)
    train_genre_count, train_count_df = count_genres(train_data_df, return_dataframe=True)

    val_count_df.to_excel("data/validation_count.xlsx", index=False)
    train_count_df.to_excel("data/train_count.xlsx", index=False)

    return train_data_df, validation_data_df

def create_binary_maps(genres):
    train_data_df = pd.read_excel("data/balanced_train_data_93.xlsx")
    validation_data_df = pd.read_excel("data/balanced_validation_data_93.xlsx")

    mlb = MultiLabelBinarizer(classes=genres)

    train_data_df["List Labels"] = train_data_df["Labels"].apply(parse_str_labels)
    validation_data_df["List Labels"] = validation_data_df["Labels"].apply(parse_str_labels)

    train_b_labels = mlb.fit_transform(train_data_df["List Labels"])

    for i, row in tqdm(train_data_df.iterrows(), total=train_data_df.shape[0]):
        ids_to_binary_labels[row["Id"]] = train_b_labels[i].tolist()

    val_b_labels = mlb.fit_transform(train_data_df["List Labels"])

    for i, row in tqdm(validation_data_df.iterrows(), total=validation_data_df.shape[0]):
        ids_to_binary_labels[row["Id"]] = val_b_labels[i].tolist()

    print("Done caching.")


def generate_aws_train_test():
    genres = load_genre_labels(file_path="data/single_genres_76.xlsx")
    data_df = pd.read_excel("data/single_genre_data_250.xlsx")

    data_df["Subtitles"] = data_df["Subtitles 1"] + data_df["Subtitles 2"]
    data_df.drop(["Subtitles 1", "Subtitles 2"], inplace=True, axis=1)

    data_chunks_df = generate_chunks(data_df, genres)

    data_chunks_df.drop(["Label"], axis=1, inplace=True)

    #train_data_chunks_df.to_csv("data/train_2.csv", index=False)
    #validation_data_chunks_df.to_csv("data/test_2.csv", index=False)


def genreate_new_gold_data():
    genres = load_genre_labels(file_path="data/all_genres.xlsx")
    new_gold_data_df = create_new_gold_data(genres, "data/new_gold_data.xlsx")
    return new_gold_data_df


def generate_aws_chunk_data():
    create_binary_maps()
    train_data_chunk_df = pd.read_csv("data/train_2.csv")
    test_data_chunk_df = pd.read_csv("data/test_2.csv")
    genres = load_genre_labels(file_path="data/new_93_genres.xlsx")

    new_columns = ["Id", "Text"]
    new_columns.extend(genres)

    train_output_data_df = pd.DataFrame(columns=new_columns)

    for i, row in tqdm(train_data_chunk_df.iterrows(), total=train_data_chunk_df.shape[0]):
        new_row = [row["Id"], row["Text"]]
        new_row.extend(ids_to_binary_labels[row["Id"]])
        df_length = train_output_data_df.shape[0]
        train_output_data_df.loc[df_length] = new_row

    test_output_data_df = pd.DataFrame(columns=new_columns)

    for i, row in tqdm(test_data_chunk_df.iterrows(), total=test_data_chunk_df.shape[0]):
        new_row = [row["Id"], row["Text"]]
        new_row.extend(ids_to_binary_labels[row["Id"]])
        df_length = test_output_data_df.shape[0]
        test_output_data_df.loc[df_length] = new_row

    train_output_data_df.to_csv("data/train_data_chunked.csv")
    test_output_data_df.to_csv("data/test_data_chunked.csv")


def generate_aws_chunk_data_numpy():
    genres = load_genre_labels(file_path="data/new_93_genres.xlsx")
    create_binary_maps(genres)
    train_data_chunk_df = pd.read_excel("data/train_cd.xlsx")
    test_data_chunk_df = pd.read_excel("data/test_cd.xlsx")

    new_columns = ["Id", "Text"]
    new_columns.extend(genres)

    train_output_data_df = pd.DataFrame(columns=new_columns)
    test_output_data_df = pd.DataFrame(columns=new_columns)

    for genre in genres:
        train_data_chunk_df[genre] = 0.0
        test_data_chunk_df[genre] = 0.0

    train_output_data_df["Binary Genres"] = train_data_chunk_df["Id"].map(ids_to_binary_labels)
    train_output_data_df["Binary Genres"].dropna(inplace=True)

    train_binary_df = pd.DataFrame(train_output_data_df["Binary Genres"].values.tolist(), columns=genres)
    #train_binary_df.to_excel("data/train_binary.xlsx", index=False)
    train_data_chunk_df.iloc[:,2:] =  train_binary_df
    train_data_chunk_df = train_data_chunk_df[new_columns]

    test_output_data_df["Binary Genres"] = test_data_chunk_df["Id"].map(ids_to_binary_labels)
    test_binary_df = pd.DataFrame(test_output_data_df["Binary Genres"].values.tolist(), columns=genres)
    #test_binary_df.to_excel("data/test_binary.xlsx", index=False)
    test_data_chunk_df.iloc[:,2:] =  test_binary_df
    test_data_chunk_df = test_data_chunk_df[new_columns]

    train_data_chunk_df.to_csv("data/train.csv", index=False)
    test_data_chunk_df.to_csv("data/test.csv", index=False)

    train_data_chunk_df.to_excel("data/train_20k.xlsx", index=False)
    test_data_chunk_df.to_excel("data/test_20k.xlsx", index=False)


if __name__ == "__main__":

    #new_gold_data_df = genreate_new_gold_data()

    #new_gold_data_df = pd.read_excel("data/new_gold_data.xlsx")

    #train_data_df, validation_data_df = stratify_split(new_gold_data_df)

    #generate_aws_train_test()

    generate_aws_chunk_data_numpy()

    #validation_data_df.to_excel("data/balanced_validation_data_100.xlsx", index=False)
    #train_data_df.to_excel("data/balanced_train_data_100.xlsx", index=False)

