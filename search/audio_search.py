import os
import shutil
from pydub import AudioSegment
from pydub.utils import make_chunks
import nltk
import json
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')

from sentence_transformers import SentenceTransformer, util

import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from search.semantic_search import *



class SemanticSearchForAudio:
    """
    Semantic search for audio class.
    """

    def __init__(self, transcription, model=None, tokenizer=None):
        """
        Initialize the SemanticSearchForAudio class.

        Args:
            transcription (str): Transcription of the audio.
            model: SentenceTransformer model for generating embeddings.
            tokenizer: Tokenizer for preprocessing text.
        """
        self.transcription = transcription
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """
        Perform mean pooling on the model output.

        Args:
            model_output: Model output containing token embeddings.
            attention_mask: Attention mask for input.

        Returns:
            torch.Tensor: Mean pooled embeddings.
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

    def get_similarity(self, query):
        """
        Get similarity between the query and audio transcription.

        Args:
            query (str): Query text.

        Returns:
            float: Similarity score between the query and audio transcription.
        """
        query_embedding = self.get_embeddings([query]).cpu().detach().numpy()
        transcription_embedding = self.get_embeddings([self.transcription]).cpu().detach().numpy()

        return round(float(util.pytorch_cos_sim(transcription_embedding, query_embedding)), 3)


class spech_search:
    """
    A class for transcribing speech from audio files using the Wav2Vec2 model.

    Attributes:
        file_path (str): Path to the audio file for transcription.
        _wav2vec_model: Wav2Vec2 model for speech transcription.
        _wav2vec_tokenizer: Tokenizer for the Wav2Vec2 model.
    """

    def __init__(self, file_path, wav2vec_model=None, wav2vec_tokenizer=None):
        """
        Initializes the SpeechTranscriber object.

        Args:
            file_path (str): Path to the audio file for transcription.
            wav2vec_model: Pretrained Wav2Vec2 model for speech transcription.
            wav2vec_tokenizer: Pretrained tokenizer for the Wav2Vec2 model.
        """
        self.file_path = file_path
        self._wav2vec_model = wav2vec_model
        self._wav2vec_tokenizer = wav2vec_tokenizer

    @property
    def wav2vec_model(self):
        """
        Property for accessing the Wav2Vec2 model for speech transcription.
        If not already loaded, it loads the pretrained model.

        Returns:
            Wav2Vec2ForCTC: Pretrained Wav2Vec2 model.
        """
        if not self._wav2vec_model:
            print("Loading Wav2Vec2 Model...")
            self._wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        return self._wav2vec_model

    @property
    def wav2vec_tokenizer(self):
        """
        Property for accessing the tokenizer for the Wav2Vec2 model.
        If not already loaded, it loads the pretrained tokenizer.

        Returns:
            Wav2Vec2Processor: Pretrained Wav2Vec2 tokenizer.
        """
        if not self._wav2vec_tokenizer:
            print("Loading Wav2Vec2 Tokenizer...")
            self._wav2vec_tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        return self._wav2vec_tokenizer

    def get_transcription(self, file_path=None):
        """
        Transcribes speech from an audio file and returns the transcription.

        Args:
            file_path (str, optional): Path to the audio file for transcription.
                If not provided, uses the file path specified during object initialization.

        Returns:
            str: Transcription of the speech from the audio file.
        """
        if not file_path:
            file_path = self.file_path

        # Load the audio and sample rate
        speech, rate = librosa.load(file_path, sr=16000)

        # Tokenize the audio and obtain input values
        input_values = self.wav2vec_tokenizer(speech, return_tensors='pt').input_values

        # Get the logits (non-normalized predictions) from the model
        with torch.no_grad():
            logits = self.wav2vec_model(input_values).logits

        # Get the predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the audio to generate text by passing the prediction to the tokenizer's decode method
        transcription = (self.wav2vec_tokenizer.batch_decode(predicted_ids)[0]).lower()

        return transcription



class SpeechSearchThroughDirectory:
    """
    A class for searching through a directory of audio files for specific queries.

    Attributes:
        audio_path (str): Path to the directory containing audio files.
        directoryPath (str): Path to the directory where intermediate files will be stored.
        saving_path (str): Path to the directory where results will be saved.
        current_path (str): Path to the current working directory.
        wav2vec_model: Wav2Vec model for audio processing.
        wav2vec_tokenizer: Tokenizer for Wav2Vec model.
        semantic_model: Semantic search model.
        semantic_tokonizer: Tokenizer for semantic search.
    """

    def __init__(self, audio_path=None, directoryPath=None, saving_path=None, current_path=None, wav2vec_model=None,
                 wav2vec_tokenizer=None, semantic_model=None, semantic_tokonizer=None, f=10, i=0):
        """
        Initializes the SpeechSearchThroughDirectory object.

        Args:
            audio_path (str, optional): Path to the directory containing audio files.
            directoryPath (str, optional): Path to the directory where intermediate files will be stored.
            saving_path (str, optional): Path to the directory where results will be saved.
            current_path (str, optional): Path to the current working directory.
            wav2vec_model: Wav2Vec model for audio processing.
            wav2vec_tokenizer: Tokenizer for Wav2Vec model.
            semantic_model: Semantic search model.
            semantic_tokonizer: Tokenizer for semantic search.
            f (int, optional): Number of processes to divide data into.
            i (int, optional): Index of the current process.
        """
        if not saving_path:
            self.saving_path = os.getcwd()
        else:
            self.saving_path = saving_path

        if not directoryPath:
            directoryPath = os.getcwd()

        if not directoryPath.endswith('/'):
            directoryPath = directoryPath + '/'

        if current_path:
            if not current_path.endswith('/'):
                current_path = current_path + '/'

        if audio_path:
            if not audio_path.endswith('/'):
                audio_path = audio_path + '/'

        self.audio_path = audio_path
        self.directoryPath = directoryPath
        self.corrubted = []
        self.processed_audios_names = []

        self.wav2vec_model = wav2vec_model
        self.wav2vec_tokenizer = wav2vec_tokenizer
        self.semantic_model = semantic_model
        self.semantic_tokonizer = semantic_tokonizer

        if not current_path:
            self.current_path = os.getcwd()
        else:
            self.current_path = current_path

        all_files = os.listdir(audio_path)
        original_wav_files = [file for file in all_files if file.endswith('.wav')]

        processed_audios_names = os.listdir(audio_path + 'parts')
        processed_audios_names = [f.replace('_scores', '').replace('csv', 'wav') for f in processed_audios_names if
                                  f.endswith('csv')]

        processed_audios_names2 = []
        try:
            shutil.copy(os.path.join(audio_path, "configurations/processed_audio_files.txt"), self.current_path)
            processed_files = open(os.path.join(self.current_path, "processed_audio_files.txt"), "r+")
            processed_audios_names2 = processed_files.readlines()
            processed_files.close()
            processed_audios_names2 = [file[:-1] for file in processed_audios_names2]
            os.remove(os.path.join(self.current_path, "processed_audio_files.txt"))
        except:
            pass

        processed_audios_names.extend(processed_audios_names2)
        original_wav_files = [file for file in original_wav_files if file not in processed_audios_names]
        original_wav_files.sort()

        n1 = int((len(original_wav_files) / f)) * i
        n2 = int((len(original_wav_files) / f)) * (i + 1)

        num = 10

        if n2 > n1 + num:
            n2 = n1 + num

        original_wav_files = original_wav_files[n1:n2]

        all_wav_files = [file for file in os.listdir(self.directoryPath) if file.endswith('.wav')]
        if len(all_wav_files) <= num:
            for i, file in enumerate(original_wav_files):
                try:
                    shutil.copy(f"{audio_path}{file}", self.directoryPath)
                    print(f"Moving {i}. {file} to data")
                except Exception as e:
                    print(f"Error while moving file to data: {e}")

        all_wav_files = [file for file in os.listdir(self.directoryPath) if file.endswith('.wav')]

        processed_audios_names3 = os.listdir(self.directoryPath + 'parts')
        processed_audios_names3 = [f.replace('_scores', '').replace('csv', 'wav') for f in processed_audios_names3 if
                                   f.endswith('csv')]

        processed_audios_names.extend(processed_audios_names3)

        for file in all_wav_files:
            if file in processed_audios_names:
                try:
                    os.remove(self.directoryPath + file)
                    print(f"Removing {self.directoryPath}{file}")
                except:
                    print(f"Can't remove {file} from {self.directoryPath}")

        self.all_wav_files = [file for file in all_wav_files if file not in processed_audios_names]

    def convert_format(self, input_file_name, input_file_format, saving_path=None):
        """
        Converts an audio file to WAV format.

        Args:
            input_file_name (str): Name of the input audio file.
            input_file_format (str): Format of the input audio file.
            saving_path (str, optional): Path to save the converted WAV file.

        Returns:
            None
        """
        input_file_path = os.path.join(self.directoryPath, input_file_name)
        m4a_audio = AudioSegment.from_file(input_file_path, format=input_file_format)
        wav_name = input_file_name.split("/")[-1].split(".")[0] + ".wav"

        if not saving_path:
            saving_path = self.directoryPath

        saving_wav_name = os.path.join(saving_path, wav_name)

        wav_exist = os.path.exists(saving_wav_name)
        if not wav_exist:
            m4a_audio.export(saving_wav_name, format="wav")
        else:
            print("WAV file already exists")

    def get_segments_with_high_score(self, file_name, folder_path=None, saving_folder=None, queries=['books to read'],
                                     thresh_score=.06, audio_length=30):
        """
        Extracts segments from an audio file that match specified queries and have a high similarity score.

        Args:
            file_name (str): Name of the input audio file.
            folder_path (str, optional): Path to the directory containing audio files.
            saving_folder (str, optional): Path to the directory where results will be saved.
            queries (list, optional): List of queries to search for in the audio.
            thresh_score (float, optional): Minimum similarity score for a segment to be saved.
            audio_length (int, optional): Length of audio segments in seconds.

        Returns:
            None
        """
        if not folder_path:
            folder_path = self.directoryPath

        if not saving_folder:
            saving_folder = self.saving_path

        seconds = audio_length

        file_with_path = os.path.join(folder_path, file_name)
        print(file_with_path)
        myaudio = AudioSegment.from_file(file_with_path, "wav")
        chunk_length_ms = seconds * 1000
        chunks = make_chunks(myaudio, chunk_length_ms)

        try:
            os.makedirs(saving_folder)
        except:
            pass

        try:
            os.makedirs(os.path.join(folder_path, 'parts'))
        except:
            pass

        try:
            os.makedirs(os.path.join(self.current_path, 'parts'))
        except:
            pass

        try:
            os.makedirs(os.path.join(saving_folder, 'result'))
        except:
            pass

        chunks_scores = {"chunks_names": []}
        for i, chunk in enumerate(chunks):
            chunk_exist = None
            chunk_full_name = file_name.split(".")[0] + f"_{i:03}.wav"

            chunck_saving = os.path.join(folder_path, 'parts/')

            try:
                os.makedirs(chunck_saving)
            except:
                pass

            chunk_name = chunck_saving + chunk_full_name

            chunk_name_for_analysis = os.path.join(self.current_path, 'parts/') + chunk_full_name
            print("Exporting", chunk_full_name)

            try:
                chunk_exist = os.path.exists(chunk_name_for_analysis)
                if chunk_exist:
                    continue
                chunk.export(chunk_name_for_analysis, format="wav")
            except:
                print("Error while exporting chunk")

            if chunk_exist:
                continue

            search = spech_search(file_path=chunk_name_for_analysis, wav2vec_model=self.wav2vec_model,
                                  wav2vec_tokenizer=self.wav2vec_tokenizer)
            transcription = search.get_transcription()
            semantic_search_via_audio = SemanticSearchForAudio(transcription, model=self.semantic_model,
                                                                  tokenizer=self.semantic_tokonizer)
            os.remove(chunk_name_for_analysis)

            chunks_scores["chunks_names"].append(chunk_full_name)
            for query in queries:
                saving_folder_for_queries = os.path.join(saving_folder, 'result')

                saving_query = os.path.join(saving_folder_for_queries, query)
                try:
                    os.makedirs(saving_query)
                    print(f"Creating saving folder for:   {query}")
                except:
                    pass

                score = semantic_search_via_audio.get_similarity(query)
                print(f"The score of the {i} audio for {query} is:  {score}")

                try:
                    x = chunks_scores[query + " scores"]
                except:
                    chunks_scores[query + " scores"] = []

                chunks_scores[query + " scores"].append(round(score + .05, 3))

                if score >= thresh_score:
                    chunk_name = saving_query + '/' + file_name.split(".")[0] + f"_{i:03}.wav"
                    print(f"{chunk_name} has {score} score")

                    try:
                        chunk.export(chunk_name, format="wav")
                    except:
                        pass

        chunks_scores_dataframe = pd.DataFrame.from_dict(chunks_scores)
        chunks_scores_saving_name = chunck_saving + file_name.split(".")[0] + "_scores.csv"
        print(chunks_scores_dataframe)
        print(f"Chunks_scores_saving_name:   {chunks_scores_saving_name}")
        chunks_scores_dataframe.to_csv(chunks_scores_saving_name, index=True)


    def after_process(self):
        """
        Remove processed audio files from the directory.

        Returns:
            None
        """
        for i, file in enumerate(self.all_wav_files):
            os.remove(f"{self.directoryPath}{file}")
            print(f"Removing {i}. {file} from data")


    def run_through_all_files(self, running_path=None, queries=['read books', 'watch movies']):
        """
        Process all audio files in the directory, extracting segments that match specified queries.

        Args:
            running_path (str, optional): Path to the current working directory.
            queries (list, optional): List of queries to search for in the audio.

        Returns:
            None
        """
        running_path = self.current_path

        try:
            os.remove(os.path.join(os.getcwd(), "config.txt"))
        except:
            pass

        try:
            shutil.copy(os.path.join(self.directoryPath, "configurations/config.txt"), running_path)
            config = json.load(open(os.path.join(running_path, "config.txt")))
            queries = config['queries']
            thresh_score = float(config['score_threshold'])
            audio_length = config['audio_length']
            print(f"Loaded queries:\n{queries}\nThreshold: {thresh_score}\nAudio length: {audio_length}")
        except:
            queries = ["books to read", "movies to watch"]
            thresh_score = 0.11
            audio_length = 30
            print(f"Using default queries:\n{queries}\nThreshold: {thresh_score}\nAudio length: {audio_length}")

        thresh_score -= .05

        for each_file in self.all_wav_files:
            self.get_segments_with_high_score(each_file, self.directoryPath, self.saving_path, queries=queries,
                                              thresh_score=thresh_score, audio_length=audio_length)

            try:
                os.remove(os.path.join(self.directoryPath, each_file))
                print(f"Removing {each_file} after processing")
            except:
                print(f"Can't remove {each_file} after processing")

        try:
            os.remove(os.path.join(os.getcwd(), "config.txt"))
        except:
            pass
