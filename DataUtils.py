import numpy as np
from sumy.nlp.stemmers import Stemmer

from EntropyUtils import *
import pandas as pd
from orderedset import *
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import glob
import os

def load_genre_labels(file_path):
    current_genres_df = None
    if file_path.endswith("csv"):
        current_genres_df = pd.read_csv(file_path)
    elif file_path.endswith("xlsx"):
        current_genres_df = pd.read_excel(file_path)

    genres = current_genres_df["Genre"]
    return genres.tolist()

def parse_str_labels(str_labels):

    if type(str_labels) != str:
        return ""

    labels = list(map(str.strip, str_labels.split(",")))
    return labels

def remove_unused_genres(genre_list, genres):
    if (type(genre_list) == str):
        genre_list = parse_str_labels(genre_list)
    genres_set = OrderedSet(list(genres))
    genre_list_set = OrderedSet(list(genre_list))
    output = list(genre_list_set.intersection(genres_set))
    return output

def generate_subtitle_files(data_df, data_type):
    for i, row in data_df.iterrows():
        f= open("data/subtitles/"+data_type+"/"+row["Id"]+".txt","w+")
        subtitles = f.write(row["Subtitles"])
        f.close()


def populate_predictions(data_df, genres=None):

    index_to_name = {}
    name_to_index = {}

    for i, genre in enumerate(genres):
        index_to_name[i] = genre
        name_to_index[genre] = i


    text_predictions = []

    for i, row in data_df.iterrows():
        prob_predictions = row[genres]
        predictions = []

        for i, pred in enumerate(prob_predictions):
            if pred >= 0.50:
                predictions.append(index_to_name[i])

        str_predictions = str(predictions).replace("[", "").replace("]", "").replace("'", "")
        text_predictions.append(str_predictions)

    data_df["Predictions"] = text_predictions

    columns = ["Id","Labels","Predictions"]
    columns.extend(genres)
    data_df = data_df[columns]
    return data_df

def condense(data_df, genres=None, prediction_column=None, label_column=None):
    text = ""
    prev_id = ""
    binary_labels_rows = None
    prediction_set = OrderedSet()
    label_set = OrderedSet()
    texts = []
    ids = []
    rows = []
    predictions = []
    labels = []

    data_df = data_df.set_index("Id")
    data_df["Id"] = data_df.index

    for i, row in data_df.iterrows():
        id = row["Id"]
        if id == prev_id and i != data_df.shape[0]-1 or i == 0:
            text = text + " " + row["Text"]
            if prediction_column is not None and pd.notna(row[prediction_column]):
                set_labels(prediction_column, prediction_set, row)

            if label_column is not None and pd.notna(row[label_column]):
                set_labels(label_column, label_set, row)

            if binary_labels_rows is not None:
                binary_labels_rows = row[genres] + binary_labels_rows
            else:
                binary_labels_rows = row[genres]
        else:
            texts.append(text)
            ids.append(prev_id)
            rows.append(binary_labels_rows)

            if prediction_column is not None:
                predictions_str = str(list(prediction_set)).replace("[","").replace("]","").replace("'","")

                predictions.append(predictions_str)

            if label_column is not None:
                labels_str = str(list(label_set)).replace("[", "").replace("]", "").replace("'", "")
                labels.append(labels_str)

            text = row["Text"]

            if prediction_column is not None and pd.notna(row[prediction_column]):
                prediction_set.clear()
                set_labels(prediction_column, prediction_set, row)
            else:
                if prediction_column is not None:
                    prediction_set.clear()

            if label_column is not None and pd.notna(row[label_column]):
                label_set.clear()
                set_labels(label_column, label_set, row)
            else:
                if label_column is not None:
                    label_set.clear()

        prev_id = id


    #columns = ["Id","Text"]
    #if label_column is not None:
    #    columns.extend([label_column])

    #if prediction_column is not None:
    #    columns.extend([prediction_column])

    #columns.extend(genres)
    new_data_df = pd.DataFrame(rows, columns=genres)
    new_data_df["Id"] = ids
    new_data_df["Text"] = texts

    if label_column is not None:
        new_data_df[label_column] = labels

    if prediction_column is not None:
        new_data_df[prediction_column] = predictions

    return new_data_df

def set_labels(label_column, label_set, row):
    label_list = list(map(str.strip, row[label_column].split(",")))
    for label in label_list:
        if label != '':
            label_set.add(label)
    return label_set

def split_string_list(genres):
    if type(genres) == str:
        if genres == "":
            return []
        genres_list = genres.split(",")
        genres_list = [i.strip() for i in genres_list]
    else:
        return ""
    return genres_list

def get_summary(text):
    LANGUAGE = "english"
    SENTENCES_COUNT = 20

    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    text = text.replace("...",".")
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    text = ""
    
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        text = text +" "+str(sentence)
    
    return text

def wordsplit(atext):
    if type(atext) == str:
        punctuation = '.,():-—;"!?•$%@“”#<>+=/[]*^\'{}_■~\\|«»©&~`£·'
        atext = atext.replace('-', ' ')
        # we replace hyphens with spaces because it seems probable that for this purpose
        # we want to count hyphen-divided phrases as separate words
        awordseq = [x.strip(punctuation).lower() for x in atext.split()]
        return awordseq


def get_chunks(atext, chunk_size=512):
    if type(atext) == str:
        wordseq = wordsplit(atext)

        # we count types and tokens in the full sequence

        seqlen = len(wordseq)
        chunks = []
        # Now we iterate through chunks
        overrun = True


        for startposition in range(0, seqlen, chunk_size):
            endposition = startposition + chunk_size

            # If this (final) chunk would overrun the end of the sequence,
            # we adjust it so that it fits, and overlaps with the previous
            # chunk.

            if endposition >= seqlen:
                if endposition > seqlen:
                    overrun = True

                endposition = seqlen
                startposition = endposition - chunk_size

                if startposition < 0:
                    #print('In at least one document, chunk size exceeds doc size.')
                    startposition = 0

            thischunk = wordseq[startposition: endposition]

            chunks.append(thischunk)

        return chunks
    else:
        return []

def get_chunks_and_measures(atext, chunk_size=512):
    wordseq = wordsplit(atext)

    # we count types and tokens in the full sequence

    overalltypect = len(set(wordseq))
    seqlen = len(wordseq)
    chunks = []
    # Now we iterate through chunks
    overrun = True
    
    ttr_list = []
    conditional_entropy_list = []
    normalized_entropy_list = []
    cumulative_sequence = []
    
    for startposition in range(0, seqlen, chunk_size):
        endposition = startposition + chunk_size

        # If this (final) chunk would overrun the end of the sequence,
        # we adjust it so that it fits, and overlaps with the previous
        # chunk.
        
        if endposition >= seqlen:
            if endposition > seqlen:
                overrun = True

            endposition = seqlen
            startposition = endposition - chunk_size

            if startposition < 0:
                print ('In at least one document, chunk size exceeds doc size.')
                startposition = 0

        thischunk = wordseq[startposition: endposition]
        
        ttr, conditional_entropy, normalized_entropy = get_all_measures(thischunk)
        
        chunks.append(thischunk)
        ttr_list.append(ttr)
        conditional_entropy_list.append(conditional_entropy)
        normalized_entropy_list.append(normalized_entropy)
        
        if not overrun:
            cumulative_text = wordseq[0: endposition]
            cumTTR, cumconditional, cumnormalized = get_all_measures(cumulative_text)
            cumulative_dict = dict()
            cumulative_dict['ttr'] = cumTTR
            cumulative_dict['conditional'] = cumconditional
            cumulative_dict['normalized'] = cumnormalized
            cumulative_sequence.append(cumulative_dict)
        
        #ttr = sum(ttr_list) / len(ttr_list)
        #conditional_entropy = sum(conditional_entropy_list) / len(conditional_entropy_list)
        #normalized_entropy = sum(normalized_entropy_list) / len(normalized_entropy_list)
    
    return chunks, ttr_list, conditional_entropy_list, normalized_entropy_list 


def load_data(file_path, genres, rows=None, validation_split=None, create_validation=False):
    data_df = pd.read_excel(file_path, nrows=rows)
    data_df = data_df[data_df.Exclude == False].reset_index(drop=True)
    data_df.replace(r'[^\x00-\x7F]+',' ', inplace=True, regex=True)
    data_df.replace(0,"", inplace=True)
    filtered_data_df = data_df
    filtered_data_df["Subtitles 1"].fillna("", inplace=True)
    filtered_data_df["Subtitles 2"].fillna("", inplace=True)
    filtered_data_df["Subtitles"] = filtered_data_df["Subtitles 1"] + filtered_data_df["Subtitles 2"]
    filtered_data_df = data_df[data_df.Subtitles.notna()].reset_index(drop=True)
    filtered_data_df.drop(["Subtitles 1", "Subtitles 2"], inplace=True, axis=1)

    if validation_split is not None:
        filtered_data_df = filtered_data_df[filtered_data_df.Training == True].reset_index()
        training_df = filtered_data_df[:int(1 - filtered_data_df.shape[0]*validation_split)].reset_index(drop=True)
        validation_df = filtered_data_df[int(1 - filtered_data_df.shape[0]*validation_split)+1: - 1].reset_index(drop=True)
    else:
        training_df = filtered_data_df[filtered_data_df.Training == True].reset_index(drop=True)
        if create_validation:
            validation_df = filtered_data_df[filtered_data_df.Validation == True].reset_index(drop=True)
    
    
    #training_df["Labels"] = training_df["Genres"].apply(parse_str_labels).tolist()
    training_df["Labels"] = training_df["Genres"].apply(remove_unused_genres, args = (genres,)).tolist()
    
    #validation_df["Labels"] = validation_df["Genres"].apply(parse_str_labels)
    if create_validation:
        validation_df["Labels"] = validation_df["Genres"].apply(remove_unused_genres, args=(genres,))
    
    training_df = training_df[training_df["Labels"].map(lambda d: len(d)) > 0].reset_index(drop=True)

    if create_validation:
        validation_df = validation_df[validation_df["Labels"].map(lambda d: len(d)) > 0].reset_index(drop=True)
    
    mlb = MultiLabelBinarizer(classes=genres)

    training_binary_labels = mlb.fit_transform(training_df["Labels"])
    training_labels =  training_binary_labels

    if create_validation:
        validation_binary_labels = mlb.fit_transform(validation_df["Labels"])
        validation_labels =  validation_binary_labels
        return training_df[["Id", "Subtitles","Labels"]], validation_df[["Id","Subtitles","Labels"]], training_labels, validation_labels #training_df, validation_df
    else:
        return training_df[["Id", "Subtitles", "Labels"]]

def generate_summary(data_df, data_type):
    summary_text = []
    rows = []
    for i, row in data_df.iterrows():
        print("Row",i)
        try:
            text = get_summary(row["Subtitles"])
            summary_text.append(text)
            rows.append(row)
        except:
            continue
    data_df = pd.DataFrame(rows)
    data_df["Summary"] = summary_text
    return data_df


def generate_aws_data_format(data_df, genres, fit=True, text_column="Chunks"):
    
    columns = data_df.columns
    columns.extend(genres)
   
    output_data_df = pd.DataFrame(columns=columns)
    mlb = MultiLabelBinarizer(classes=genres)

    b_labels = mlb.fit_transform(data_df["Labels"])


    for i, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        new_row = [row["Id"], row[text_column]]
        new_row.extend(b_labels[i].tolist())
        df_length = output_data_df.shape[0]
        output_data_df.loc[df_length] = new_row
    return output_data_df


def generate_binary_columns(data_df, genres, label_column="List Labels"):
    columns = data_df.columns.tolist()
    columns.extend(genres)

    output_data_df = pd.DataFrame(columns=columns)
    mlb = MultiLabelBinarizer(classes=genres)

    b_labels = mlb.fit_transform(data_df[label_column])

    i = 0
    for id, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        new_row = [row[column] for column in data_df.columns]
        new_row.extend(b_labels[i].tolist())
        df_length = output_data_df.shape[0]
        output_data_df.loc[df_length] = new_row
        i += 1
    return output_data_df

    
def generate_chunks(data_df, genres, chunk_size=512, return_string=True, output_labels = True, label_column="Labels", text_column="Subtitles", include_binary_genre_columns=True):
    rows = []

    if output_labels:
        columns = ["Id", "Text", label_column]
        columns.extend(genres)
        data_df = data_df[columns]
    else:
        columns = ["Id", "Text"]

    output_data_df = pd.DataFrame(rows, columns=columns)

    for i, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        chunks = get_chunks(row[text_column], chunk_size=chunk_size)
        for i, chunk_text in enumerate(chunks):

            if output_labels:
                new_row = [row["Id"], chunk_text, row[label_column]]
            else:
                new_row = [row["Id"], chunk_text]

            new_row.extend(row[genres].values.tolist())

            df_length = output_data_df.shape[0]
            output_data_df.loc[df_length] = new_row

    if return_string:
        output_data_df["Text"] = output_data_df["Text"].apply(lambda x: str(x).replace("[", "").replace("]","").replace("'","").replace(",","").replace('"',''))
    return output_data_df


def get_labels(binary_labels, genres, return_string=True):
    indexes = np.nonzero(binary_labels)[0]
    labels = []
    for i in indexes:
        labels.append(genres[i])
    if return_string:
        str_labels = str(labels).replace("[","").replace("]","").replace("'","")
        return str_labels
    else:
        return labels


def create_comma_separated_str_values(predictions_df, genres):
    predictions_or_labels = []
    for i, row in predictions_df.iterrows():
        binary_predictions = row[genres].tolist()
        predictions_str = get_labels(binary_predictions, genres)
        predictions_or_labels.append(predictions_str)
    return predictions_or_labels

def merge_files(base_path, output_file_name, export=True, columns=None):
    # files = glob.glob(base_path + '*.xlsx')
    files = sorted(glob.glob(base_path + "episode_vectors_*.xlsx"), key=os.path.getmtime)

    all_data = pd.DataFrame(columns=columns)
    for i, filename in enumerate(files):
        dataframe_i = pd.read_excel(filename)
        all_data = all_data.append(dataframe_i, ignore_index=True)

    if export:
        all_data.to_excel(base_path + output_file_name, index=False)

    return all_data

def convert_binary(value):
    if type(value) == str:
        return value

    if value >= 0.50:
        return 1
    else:
        return 0

if __name__ == "__main__":

    columns = ["Id", "action", "horror", "thriller", "crime", "romance", "romantic comedy", "documentary", "science fiction", "fantasy", "lgbt-related",
               "musical", "biographical", "adventure", "war", "mystery", "teen", "childrens", "western", "coming-of-age story", "martial arts", "silent",
               "christmas", "noir", "buddy", "slasher", "historical", "heist", "erotic", "monster", "zombie", "spy", "neo-noir", "vampire", "dystopian",
               "crime thriller", "sports", "melodrama", "prison", "comic science fiction", "disaster", "family", "post-apocalyptic", "parody",
               "speculative fiction", "superhero", "erotic thriller", "political thriller", "psychological thriller"]

    merge_files("../data/", "pluto_vectors_2.xlsx", columns=columns)


