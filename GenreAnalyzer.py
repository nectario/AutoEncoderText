from fuzzywuzzy import fuzz as fz
from collections import Counter
from scipy.spatial.distance import cdist
from nltk.stem import WordNetLemmatizer
import configparser
from common.utils import *

#import pywikibot
#from pywikibot import pagegenerators as pg
import pickle
exclude_genres =["acid western","philosophical fiction","social science fiction","humor","clay animation","mashup","non-fiction","beat 'em up","love comedy","science fiction animation","manga","bossa nova","classical music","biopunk","hardcore pornography","human","commedia lirica","sketch story","russian rock","comedy of intrigue","french new wave","biblical genre","soft rock","mountain","news","visual novel","epic poem","television special","blaxploitation horror","picture book","new german cinema","remake","black-and-white photographic","metafiction","fraud","bildungsroman","science fiction novel","showtune","dieselpunk","folk rock","psychological drama","action role-playing game","hack and slash","ethnofiction","cooking","high fantasy","hindustani classical music","sequel","panel game","conspiracy fiction","role-playing video game","indian soap opera","fantasy literature","psychedelic","mafia","third-person shooter","thrash metal","donghua","heroic fantasy","industrial rock","graphic adventure game","social guidance","golden age of porn","medieval fantasy","rhythm and blues","cr√≥nica","animated series","making-of","anime television program","puppetoon animation","horror television","heartland rock","sea adventure novel","new wave","epic literature","gambling","time travel","anti-humor","film adaptations of uncle tom's cabin","law","psychology","outlaw biker","horror literature","k-pop","electronic dance music","psychological experiment","persian","popular-science","hyperlink cinema","toei company","phantasmagoria","non-fiction literature","kabuki","animated documentary","reality","culture clash comedy","expressionism","mumblecore","stage and screen","electropop","adventure science fiction","caper story","surreal humour","first-person shooter","burlesque","tale","i eat your skin","sport","document","grotesque","traditional heavy metal","country music","prequel","serial","autofiction","erotic novel","adventure fiction","isekai","czechoslovak new wave","psychedelic rock","lds cinema","musicarello","poetry","shoot 'em up","feature","genre movie","novella","young adult literature","sports video game","hijacking","gourmet","epistolary novel","ethnographic","hazem","pop rock","slavic fantasy","interactive fiction","noir novel","family drama","lgbt literature","operatic pop","euro war","urban fantasy","home video","based on books","photograph","dungeons & dragons","financial thriller","detective","docu-reality","culinary art","mafia comedy","fashion show","religious satire","horror novel","legal thriller","intriga","psycho-biddy","soul music","classic rock","short story","police","power metal","creationism","christian art","bad movie"]

class GenreAnalyzer():
    """
    Class to perform genre analysis given the film/show file, for example, film_data.csv. This includes counting the number of genre tags, finding subgenres
    based on text similarity or word emeddings similarity.
    """
    def __init__(self, film_data=None, episodes_data=None, dbpedia_filepath=None, genre_colum="Genres", generate_genre_count_file=True, current_genres=None):

        film_data = film_data[film_data.Exclude == False]
        self.dbpedia_film_data_df = None

        if dbpedia_filepath is not None:
            self.dbpedia_film_data_df = pd.read_excel(dbpedia_filepath)

        self.wikidata_film_data_df = film_data
        self.wikidata_episodes_data_df = episodes_data

        self.wikidata_genre_count_df = None
        self.dbpedia_genre_count_df = None

        if self.wikidata_film_data_df is not None:
            self.wikidata_genre_count_map, self.wikidata_film_genre_count_df = self.count_genres(data_df=self.wikidata_film_data_df, genre_column=genre_colum, return_dataframe=True)

        genre_count = []
        for genre in current_genres:
            count = self.wikidata_genre_count_map[genre]
            genre_count.append(count)

        curent_genre_count_df = pd.DataFrame(columns=["Genres", "Count"])
        curent_genre_count_df["Genres"] = current_genres
        curent_genre_count_df["Count"] = genre_count
        curent_genre_count_df.to_excel("data/current_genre_count.xlsx", index=False)

        if self.wikidata_episodes_data_df is not None:
            self.wikidata_episodes_genre_count_map, self.wikidata_episodes_genre_count_df = self.count_genres(data_df=self.wikidata_episodes_data_df, genre_column=genre_colum, return_dataframe=True)

        if self.dbpedia_film_data_df is not None:
            self.dbpedia_genre_count_map, self.dbpedia_genre_count_df = self.count_genres(data_df=self.dbpedia_film_data_df, genre_column="Dbpedia_Genres", return_dataframe=True)
            if generate_genre_count_file:
                self.dbpedia_genre_count_df.to_excel("data/DBPediaGenreCount.xlsx", index=False)

        if generate_genre_count_file:
            self.wikidata_film_genre_count_df.to_excel("data/WikiDataFilmGenreCount.xlsx", index=False)
            #self.wikidata_episodes_genre_count_df.to_excel("data/WikiDataSeriesGenreCount.xlsx", index=False)

        self.glove_model = None

        #self.top_5k_film_data_doc_ids = get_top_5k_films_ids(file_path="data/document-ids.p")

        self.init_configs()

    def init_configs(self):

        self.standard_genre_config = configparser.ConfigParser()
        self.standard_genre_config.read("config/standard_genres.properties")

        self.genre_config = configparser.ConfigParser()
        self.genre_config.read("config/genres.properties")
        self.genre_categories_config = self.genre_config["Genres"]
        self.genre_mappings = {}

        for genre in self.genre_categories_config.keys():
            value = self.get_config_property(genre, config=self.genre_categories_config)
            if value is not None:
                self.genre_mappings[genre] = value

        self.exclude = self.get_config_property("exclude", config=self.genre_categories_config)

        self.lemmatizer = WordNetLemmatizer()
        self.subgenres = set()

        for main_genre, subgenres in self.genre_categories_config.items():
            for subgenre in subgenres:
                self.subgenres.add(subgenre)

    def get_config_property(self, property, config=None):
        try:
            output_value = config[property].split(",")

            if len(output_value) == 1:
                return output_value[0].strip()

            return [value.strip() for value in output_value]
        except KeyError:
            return None

    def split_str_list(self, str_labels):
        if type(str_labels) != str:
            return []
        labels = list(map(str.strip, str_labels.split(",")))
        return labels

    def get_all_genres_from_data(self, df, genre_column=None, parse=True):
        """
        Get the list of all genres without removing the duplicates. This will be used to count the
        frequency of each genre.
        :param df:
        :param genre_colum:
        :param parse:
        :return: all genres dataframe
        """
        all_genres = []
        for i, row in df.iterrows():
            if parse:
                all_genres.extend(self.split_str_list(row[genre_column]))
            else:
                all_genres.append(row[genre_column])
        return all_genres

    def get_standard_name(self, names, config=None, return_count=False):

        standard_names = []
        counts = []

        if type(names) == list:
            values = self.cleanup(names)
            for val in values:
                standard_genre_name = self.get_config_property(val, config=config["STANDARD_GENRE_MAPPINGS"])
                if standard_genre_name != None and standard_genre_name not in exclude_genres:
                    standard_names.append(standard_genre_name)

                    count = self.wikidata_genre_count_map[val]
                    counts.append(count)
            if return_count:
                return standard_names, counts
            else:
                return standard_names
        else:
            genre = config["STANDARD_GENRE_MAPPINGS"][names].strip()
            if return_count:
                return genre.replace("[","").replace("]",""), self.wikidata_genre_count_map[genre]
            return genre.replace("[","").replace("]","")

    def populate_close_caption_data(self, gold_data_df):
        """
        Populate the subtitles on the Excel file. Note: Because Excel has a limit of around 25000
        characters in a cell, we split them in two cells.
        """
        film_dialogs_1 = []
        film_dialogs_2 = []
        file_name = ""
        for i, row in gold_data_df.iterrows():
            try:
                file_name = row["Id"] + ".txt"
                with open(r"data/open_sub_dump/film_subtitles/"+file_name, encoding="utf-8") as cc_file:
                    dialog = " ".join(cc_file.readlines()).replace("\n","").replace("="," ").replace("?. ","? ").replace("  "," ")
                    if len(dialog) > 25000:
                        film_dialogs_1.append(str(dialog[0:25000]))
                        film_dialogs_2.append(str(dialog[25000:-1]))
                    else:
                        film_dialogs_1.append(dialog)
                        film_dialogs_2.append("")
            except FileNotFoundError:
                print("File not found:",file_name)
                film_dialogs_1.append("")
                film_dialogs_2.append("")

        gold_data_df["Subtitles 1"] = film_dialogs_1
        gold_data_df["Subtitles 2"] = film_dialogs_2

        return gold_data_df

    def get_gold_data(self, data_df, genre_column=None, wikipedia_data=None):

        data_df = data_df.rename(columns={"title": "Title", "overview": "Overview", "wikidata_genres": "WikiData Genres", "main_subject": "Subject", "cc_file_path": "Id", "film_release_date":"Date"}, errors="raise")

        standard_genre_names = []
        standard_genre_counts = []
        plot_summaries = []
        ids = []
        is_top_5k = []

        data_df = data_df.fillna("")
        for i, row in data_df.iterrows():
            id = row["Id"]
            if id in self.top_5k_film_data_doc_ids:
                is_top_5k.append(True)
            else:
                is_top_5k.append(False)

            plot = None
            if id in wikipedia_data.keys():
                if wikipedia_data[id].get("Plot") is not None:
                    plot = " ".join(wikipedia_data[id].get("Plot"))
                    if len(plot) > 25000:
                        print("Found Plot larger than 25000!")
            genres = [genre for genre in str(row[genre_column]).split(",")]
            standard_names, genre_counts = self.get_standard_name(genres, config=self.standard_genre_config, return_count=True)
            ids.append(id)
            plot_summaries.append(plot)
            standard_genre_names.append(str(standard_names).replace("[","").replace("]","").replace("'","").replace('"',''))
            standard_genre_counts.append(str(genre_counts).replace("[","").replace("]",""))

        data_df["Plot"] = plot_summaries
        data_df["Genres"] = standard_genre_names
        data_df["Frequency"] = standard_genre_counts
        data_df["IsTop5K"] = is_top_5k

        data_df = data_df[data_df['WikiData Genres'] != 0]

        return data_df[["Id", "Date","Title", "Overview", "WikiData Genres", "Genres", "Frequency", "Subject", "Plot","IsTop5K"]]

    def count_genres(self, data_df=None, return_dataframe=False, genre_column="Genres", columns=["Genre", "Count"]):
        """
        Perform a frequency count on all genres.
        :return:
        """
        genre_list = self.get_all_genres_from_data(data_df, genre_column=genre_column)
        genre_list = self.cleanup(genre_list)

        genre_count_map = Counter(genre_list)
        genre_count_df = pd.DataFrame(list(genre_count_map.items()), columns=columns).sort_values(by=["Count"], ascending=False)
        #genre_count_df.replace("romantic ", "romance ", regex=True, inplace=True)

        if return_dataframe:
            return genre_count_map, genre_count_df,
        else:
            return genre_count_map

    def cleanup(self, genre_list):
        genre_list = list(map(lambda x: x.replace("film", "").strip(" ").lower(), genre_list))
        return genre_list

    def export_similar_genre_list(self, similarity_ratio=80, filename="data/genre/genre_outliers.xlsx"):
        genre_count_map = self.wikidata_genre_count_map

        source_genres = []
        source_genres_counts = []
        target_genres = []
        target_genres_counts = []
        similarity_ratios = []
        already_compared = set()

        for i, (source_genre, source_count) in enumerate(genre_count_map.items()):
            for j, (target_genre, target_count) in enumerate(genre_count_map.items()):
                if i == j:
                    continue

                ratio = self.fuzzy_similarity(source_genre, target_genre)

                if ratio >= similarity_ratio and not (i,j) in already_compared:
                    source_genres.append(source_genre)
                    source_genres_counts.append(source_count)

                    target_genres.append(target_genre)
                    target_genres_counts.append(target_count)
                    similarity_ratios.append(ratio)

                already_compared.add((i,j))
                already_compared.add((j, i))

        similar_genres_df = pd.DataFrame(columns=["Source Genre", "Target Genre", "Similarity", "Source Count", "Target Count"])
        similar_genres_df["Source Genre"] = source_genres
        similar_genres_df["Target Genre"] = target_genres
        similar_genres_df["Similarity"] = similarity_ratios
        similar_genres_df["Source Count"] = source_genres_counts
        similar_genres_df["Target Count"] = target_genres_counts

        similar_genres_df.sort_values(by="Similarity", ascending=False, inplace=True)
        similar_genres_df.to_excel(filename, index=False)

        #self.create_genre_mapping_config(genre_count_map, similar_genres_df)

        return similar_genres_df

    def create_standard_genre_mapping_config(self, similar_genres_df, genre_config_file):
        genre_map = {}
        for i, row in similar_genres_df.iterrows():
            if row["Source Count"] > row["Target Count"]:
                genre_map[row["Target Genre"]] = row["Source Genre"]
            else:
                genre_map[row["Source Genre"]] = row["Target Genre"]

        genre_list = sorted(self.wikidata_genre_count_map.keys())

        for genre in genre_list:
            if genre not in genre_map.keys():
                genre_map[genre] = genre

        genre_mapping_config = configparser.ConfigParser(genre_map)
        with open(genre_config_file, 'w', encoding='utf-8') as configfile:
            genre_mapping_config.write(configfile)

    def find_subgenres(self, use_vector_similarity=False, use_string_similarity=False, metric="cosine"):
        """
        This function will search for subgenres based on similarity measures, such as text or vector similarities.
        It will compare all genres with all other genres.

        :param use_vector_similarity:
        :param use_string_similarity:
        :return: dataframe with genre/subgenre categorization
        """
        genres = {}
        for i, i_row in self.wikidata_genre_count_df.iterrows():
            source_genre = i_row["Genre"].strip(" ")

            if source_genre in self.subgenres:
                continue

            genres[source_genre] = []
            genre_vector_similarities_df = pd.DataFrame(columns=["Source Genre", "Target Genre", "Distance", "Similarity"])
            source_genre_list = []
            target_genre_list = []
            distances = []
            similarities = []

            for j, j_row in self.wikidata_genre_count_df.iterrows():
                target_genre = j_row["Genre"].strip(" ")

                if i == j or source_genre == target_genre or target_genre in self.exclude:
                    continue

                target_genre_tokens = split(target_genre)
                target_genre_tokens = map(str.strip, target_genre_tokens)

                if use_string_similarity:
                    fuzzy_ratio = self.fuzzy_similarity(self.lemmatizer.lemmatize(source_genre), self.lemmatizer.lemmatize(target_genre))

                    if fuzzy_ratio >= 58 \
                            and target_genre not in self.exclude:
                        genres[source_genre].append(target_genre)
                        self.subgenres.add(target_genre)
                    else:
                        if len(source_genre.split(" ")[0]) != 1 and len(target_genre.split(" ")[0])  != 1:
                            for target_genre_token in target_genre_tokens:
                                if target_genre_token in source_genre \
                                        and target_genre_token != "" \
                                        and target_genre_token != "on" \
                                        and target_genre_token != "and"\
                                        and " " not in source_genre and target_genre not in self.exclude:
                                    genres[source_genre].append(target_genre)
                                    self.subgenres.add(target_genre)
                                    break

                if use_vector_similarity:
                    try:
                        distance, similarity = self.vector_similarity(source_genre.replace(" ", "-"), target_genre.replace(" ", "-"), round=False, metric=metric)
                        source_genre_list.append(source_genre)
                        target_genre_list.append(target_genre)
                        distances.append(distance)
                        similarities.append(similarity)

                        genres[source_genre].append(target_genre)
                    except KeyError:
                        continue

            if use_vector_similarity:
                genre_vector_similarities_df["Source Genre"] = source_genre_list
                genre_vector_similarities_df["Target Genre"] = target_genre_list
                genre_vector_similarities_df["Distance"] = distances
                genre_vector_similarities_df["Similarity"] = similarities

                genre_vector_similarities_df.sort_values(by="Distance", ascending=True, inplace=True)

                genres[source_genre] = genre_vector_similarities_df["Target Genre"][0:13].tolist()

            if source_genre in self.genre_mappings.keys() and not use_vector_similarity:
                if len(self.genre_mappings[source_genre]) > 1:
                    genres[source_genre].extend(self.genre_mappings[source_genre])
                else:
                    genres[source_genre].append(self.genre_mappings[source_genre])

        return pd.DataFrame({ key:pd.Series(value) for key, value in genres.items()})

    def fuzzy_similarity(self, source_genre, target_genre):
        """
        Perform fuzzy match text comparison using levenshtein distance. In addition, if first characters are not equal, reduce the weight of
        the similarity
        :param source_genre:
        :param target_genre:
        :return:
        """
        weight = 1.0
        ratio = fz.ratio(source_genre, target_genre)

        if source_genre[0] != target_genre[0] and ratio > 85:
            weight = 0.7

        if " " in source_genre and " " in target_genre:
            source_genre_tokens = source_genre.split()
            target_genre_tokens = target_genre.split()
            if len(source_genre_tokens) == len(target_genre_tokens):
                source_set = set(source_genre_tokens)
                target_set = set(target_genre_tokens)
                if source_set == target_set:
                    return 100

                ratios = []
                for i in range(len(source_genre_tokens)):
                    ratios.append(fz.ratio(source_genre_tokens[i], target_genre_tokens[i]))

                return np.mean(ratios)

        return ratio * weight

    def load_glove(self, glove_file_path):
        """
        Load the glove file
        :param glove_file_path:
        :return:
        """
        f = open(glove_file_path, encoding="utf-8")
        glove = {}
        for line in tqdm(f):
            value = line.split(" ")
            word = value[0]
            coef = np.array(value[1:], dtype='float32')
            glove[word] = coef
        f.close()
        return glove

    def vector_similarity(self, word_1, word_2, round=False, metric='cosine', glove_vector_file="data/glove/glove.42B.300d/glove.42B.300d.txt"):
        """
        Compare two genres based on Glove vector similarity.

        :param word_1:
        :param word_2:
        :param round:
        :param metric: "cosine", "euclidean" etc.
        :return: distance and similarity
        """
        if self.glove_model is None:
            self.glove_model = self.load_glove(glove_file_path=glove_vector_file)

        vector_1 = self.glove_model[word_1]
        vector_2 = self.glove_model[word_2]

        vector_1 = np.atleast_2d(vector_1)
        vector_2 = np.atleast_2d(vector_2)

        if round:
            vector_1 =  np.round(vector_1)
            vector_2 = np.round(vector_2)

        distance = cdist(vector_1, vector_2, metric=metric)
        similarity = np.squeeze(1 - distance)
        return distance, similarity


if __name__ == "__main__":

    film_data_df = pd.read_excel("data/film_data_lots.xlsx")
    #wikidata_episodes_data_df = pd.read_excel("data/open_sub_dump/episodes_data.xlsx")
    with open("data/genres.pickle", "rb") as f:
        genres = pickle.load(f)

    genre_analyzer = GenreAnalyzer(film_data=film_data_df, genre_colum="Genres", current_genres=genres)
    exit(0)

    with open('data/open_sub_dump/wikipedia_data_films', 'rb') as pickle_file:
        wikipedia_data_films = pickle.load(pickle_file)

    #with open('data/open_sub_dump/wikipedia_data_series', 'rb') as pickle_file:
    #    wikipedia_data_series = pickle.load(pickle_file)

    gold_data_films = genre_analyzer.get_gold_data(data_df=film_data_df, genre_column="WikiData Genres", wikipedia_data=wikipedia_data_films)
    #gold_data_episodes = genre_analyzer.get_gold_data(data_df=wikidata_episodes_data_df, genre_column="WikiData Genres", wikipedia_data=wikidata_episodes_data_df)

    genre_analyzer.populate_close_caption_data(gold_data_films)
    #genre_analyzer.populate_close_caption_data(gold_data_episodes)

    gold_data_films = gold_data_films.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)
    #gold_data_episodes = gold_data_episodes.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

    gold_data_films.to_excel("data/gold_data_films.xlsx", index=False, encoding="utf-8")
    #gold_data_episodes.to_excel("data/gold_data_episodes.xlsx", index=False, encoding="utf-8")

    print("Done")
