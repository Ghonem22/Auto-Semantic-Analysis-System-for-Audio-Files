import faiss
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2ForCTC

import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
import pandas as pd
import os
import glob
import warnings
import datetime
import time
import shutil
import librosa
from pydub import AudioSegment
from pydub.utils import make_chunks

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import util


warnings.filterwarnings("ignore")


class SemanticSearch:
    """
    Class for performing semantic search on a dataset.
    """

    def __init__(self, csvFilePath, model_ckpt=None, model=None, tokenizer=None):
        """
        Initialize the SemanticSearch class.

        Args:
            csvFilePath (str): Path to the CSV file.
            model_ckpt (str, optional): Checkpoint name or path for the model. Defaults to None.
            model (str, optional): Pretrained model. Defaults to None.
            tokenizer (str, optional): Pretrained tokenizer. Defaults to None.
        """
        self.csvFilePath = csvFilePath
        if model_ckpt:
            self.model_ckpt = model_ckpt
        else:
            self.model_ckpt = "sentence-transformers/all-MiniLM-L6-v2"

        self._dataset = None
        self._dataset_with_embeddings = None
        self._model = model
        self._tokenizer = tokenizer

    @property
    def dataset(self):
        """
        Get the dataset from the CSV file.

        Returns:
            datasets.Dataset: Dataset object containing the data from the CSV file.
        """
        if not self._dataset:
            print("Reading and preparing dataset...")
            df = pd.read_csv(self.csvFilePath, sep=",", header=0, encoding="unicode_escape")
            self._dataset = Dataset.from_pandas(df)
        return self._dataset

    @property
    def model(self):
        """
        Get the model for embedding generation.

        Returns:
            transformers.PreTrainedModel: Pretrained model for embedding generation.
        """
        if not self._model:
            print("Loading Model...")
            self._model = AutoModel.from_pretrained(self.model_ckpt)
        return self._model

    @property
    def tokenizer(self):
        """
        Get the tokenizer for tokenizing the text.

        Returns:
            transformers.PreTrainedTokenizer: Pretrained tokenizer for tokenizing the text.
        """
        if not self._tokenizer:
            print("Loading Tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        return self._tokenizer

    @property
    def dataset_with_embeddings(self):
        """
        Get the dataset with additional embeddings column.

        Returns:
            datasets.Dataset: Dataset object with an additional column for embeddings.
        """
        if not self._dataset_with_embeddings:
            print("Creating a new column with word embeddings of the text...")

            self._dataset_with_embeddings = self.dataset.map(
                lambda x: {"embeddings": self.get_embeddings(x["Text"]).cpu().numpy()[0]}
            )

        return self._dataset_with_embeddings.add_faiss_index(column="embeddings")

    def mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings.

        Args:
            model_output (torch.Tensor): Model output containing token embeddings.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Mean-pooled embeddings.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embeddings(self, text_list):
        """
        Get embeddings for a list of texts.

        Args:
            text_list (list): List of texts.

        Returns:
            torch.Tensor: Embeddings for the input texts.
        """
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )

        encoded_input = {k: v for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self.mean_pooling(model_output, encoded_input["attention_mask"])

    def most_similar_sentence(self, query, n=3):
        """
        Find the most similar sentences to a given query.

        Args:
            query (str): Query sentence.
            n (int, optional): Number of most similar sentences to retrieve. Defaults to 3.

        Returns:
            pandas.DataFrame: DataFrame containing the most similar sentences and their scores.
        """
        try:
            query_embedding = self.get_embeddings([query]).cpu().detach().numpy()
            scores, most_similar = self.dataset_with_embeddings.get_nearest_examples(
                "embeddings", query_embedding, k=n
            )

            del most_similar["embeddings"]

            most_similar["score"] = scores

            most_similar = pd.DataFrame(most_similar)
            most_similar.sort_values(by=['score'], ascending=False, inplace=True)
            most_similar.reset_index(drop=True, inplace=True)

            return most_similar

        except KeyError:
            print("The CSV file doesn't have a 'Text' column.")
            return False

    def _dataset_conversion(self, hugging_dataset):
        """
        Convert a HuggingFace Dataset object into a DataFrame.

        Args:
            hugging_dataset (datasets.Dataset): HuggingFace Dataset object.

        Returns:
            pandas.DataFrame: DataFrame converted from the HuggingFace Dataset.
        """
        data = {}
        for col in hugging_dataset.column_names:
            data[col] = hugging_dataset[col]

        return pd.DataFrame(data)

    def semantic_search(self, queries, sorting=None):
        """
        Perform semantic search on the dataset.

        Args:
            queries (list): List of query sentences.
            sorting (str, optional): Query sentence to sort the results by. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing the results of semantic search.
        """
        try:
            dataframe = self._dataset_conversion(self.dataset_with_embeddings)
            for query in queries:
                query_embedding = self.get_embeddings([query]).cpu().detach().numpy()

                dataframe[query] = dataframe.embeddings.apply(
                    lambda x: round(float(util.pytorch_cos_sim(x, query_embedding)), 3)
                )
            del dataframe["embeddings"]

            if sorting:
                if sorting not in queries:
                    print("Sorting should be one of the query elements.")
                else:
                    dataframe.sort_values(by=[sorting], ascending=False, inplace=True)
            return dataframe

        except KeyError:
            print("The CSV file doesn't have a 'Text' column.")
            return False


class SemanticSearchThroughDirectory:
    """
    Class for performing semantic search across multiple CSV files in a directory.
    """

    def __init__(self, directoryPath=None, saving_path=None):
        """
        Initialize the SemanticSearchThroughDirectory object.

        Args:
            directoryPath (str): Path to the directory containing the CSV files.
                                Defaults to the current working directory.
            saving_path (str): Path to the directory where the search results will be saved.
                               Defaults to the current working directory.
        """
        if not saving_path:
            self.saving_path = os.getcwd()
        else:
            self.saving_path = saving_path

        if not directoryPath:
            directoryPath = os.getcwd()

        if not directoryPath.endswith('/'):
            directoryPath = directoryPath + '/'

        self.directoryPath = directoryPath
        self.csv_files = glob.glob(self.directoryPath + '*.csv')
        self.processed_files_names = None

    def most_similar_sentence_multiple_files(self, queries, n=3):
        """
        Find the most similar sentences to the given queries in multiple CSV files.

        Args:
            queries (list): List of queries (strings) to search for.
            n (int): Number of most similar sentences to retrieve. Defaults to 3.
        """
        for file in self.csv_files:
            print("\n\n\n********** Start processing {} **********\n".format(file.split("/")[-1]))
            search = SemanticSearch(csvFilePath=file)

            for query in queries:
                out = search.most_similar_sentence(query, n)
                try:
                    if out is None:
                        print("The CSV file {} doesn't have a Text column".format(file.split("/")[-1]))
                except ValueError:
                    file_name = file.split("/")[-1].split(".")[0] + '_' + query + '.csv'
                    saving_name = os.path.join(self.saving_path, 'most_similar')
                    final_saving_name = os.path.join(saving_name, file_name)

                    isExist = os.path.exists(self.saving_path)
                    if not isExist:
                        os.makedirs(self.saving_path)
                        print("Creating saving folder")

                    isExist2 = os.path.exists(saving_name)
                    if not isExist2:
                        os.makedirs(saving_name)
                        print("Creating semantic_search folder")

                    out.to_csv(final_saving_name)

    def semantic_search_multiple_files(self, queries, sorting=None):
        """
        Perform semantic search for the given queries in multiple CSV files.

        Args:
            queries (list): List of queries (strings) to search for.
            sorting (str): Sorting method for the search results. Defaults to None.
        """
        for file in self.csv_files:
            print("\n\n\n********** Start processing {} **********\n".format(file.split("/")[-1]))
            search = SemanticSearch(csvFilePath=file)

            out = search.semantic_search(queries=queries, sorting=sorting)
            try:
                if out is None:
                    print("The CSV file {} doesn't have a Text column".format(file.split("/")[-1]))
            except ValueError:
                file_name = file.split("/")[-1].split(".")[0] + '_' + 'semantic_search_result' + '.csv'
                saving_name = os.path.join(self.saving_path, 'semantic_search')
                final_saving_name = os.path.join(saving_name, file_name)

                isExist = os.path.exists(self.saving_path)
                if not isExist:
                    os.makedirs(self.saving_path)
                    print("Creating saving folder")

                isExist2 = os.path.exists(saving_name)
                if not isExist2:
                    os.makedirs(saving_name)
                    print("Creating semantic_search folder")

                out.to_csv(final_saving_name)

    def final_processing(self, queries=None, sorting=None):
        """
        Perform the final processing of the semantic search for the given queries.

        Args:
            queries (list): List of queries (strings) to search for. Defaults to predefined queries.
            sorting (str): Sorting method for the search results. Defaults to None.
        """
        if queries is None:
            queries = ['books', 'books to read', 'read', 'films', 'movies', 'movies to watch']
        self.csv_files = glob.glob(self.directoryPath + '*.csv')

        # Remove queries file and processed file if exist
        os.remove(os.path.join(os.getcwd(), "queries.txt"))

        os.remove(os.path.join(os.getcwd(), "processed_files.txt"))

        # Get queries if exist
        shutil.copy(os.path.join(self.directoryPath, "queries.txt"), os.getcwd())
        queries_file = open(os.path.join(os.getcwd(), "queries.txt"), "r+")
        queries_data = queries_file.readlines()
        queries_file.close()
