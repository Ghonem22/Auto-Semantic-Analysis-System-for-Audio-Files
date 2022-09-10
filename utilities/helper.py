from sentence_transformers import SentenceTransformer, util
import torch
import warnings
import librosa
warnings.filterwarnings("ignore")
import re
import string

from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('punkt')

from nltk.tokenize import word_tokenize



class semantic_search_for_audio:
    def __init__(self, transcription, model=None, tokenizer=None):
        self.transcription = transcription

        self.model = model
        self.tokenizer = tokenizer

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embeddings(self, text_list):
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )

        encoded_input = {k: v for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self.mean_pooling(model_output, encoded_input["attention_mask"])

    def get_similarity(self, query):
        query_embedding = self.get_embeddings([query]).cpu().detach().numpy()
        transcription_embedding = self.get_embeddings([self.transcription]).cpu().detach().numpy()

        return round(float(util.pytorch_cos_sim(transcription_embedding, query_embedding)), 3)


class spech_search:
    def __init__(self, file_path, wav2vec_model=None, wav2vec_tokenizer=None):
        self.file_path = file_path
        self.wav2vec_model = wav2vec_model
        self.wav2vec_tokenizer = wav2vec_tokenizer

    def preprocess_text(self, sen):

        english_punctuations = string.punctuation
        punctuations_list = english_punctuations

        text = sen

        # lower case
        def lower_case(text):
            res = ''
            for ligne in text:
                res = res + ligne.lower()
            return res

        text = lower_case(text)

        # remove urls
        def remove_url(res):
            res = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', res, flags=re.MULTILINE)
            return res

        text = remove_url(text)

        # remove punctuations
        def remove_punctuation(res):
            translator = str.maketrans('', '', punctuations_list)
            res = res.translate(translator)
            return res

        text = remove_punctuation(text)

        # remove single chars
        def remove_single_car(res):
            res = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', res)

            return res

        text = remove_single_car(text)

        # #remove repeating chars
        # def remove_repeating_char(res):
        #     res= re.sub(r'(.)\1', r' ', res)
        #     return res

        # text = remove_repeating_char(text)

        # remove numbers // must be before one character
        def remove_numbers(res):
            regex = re.compile(r"(\d|[\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669])+")
            res = re.sub(regex, ' ', res)

            return res

        text = remove_numbers(text)

        # remove single chars

        text = remove_single_car(text)

        # remove_extra_whitespace(string):
        def remove_white_space(res):
            res = re.sub(' +', ' ', res)
            return res

        text = remove_white_space(text)

        def remove_stopwords(text):
            text_tokens = word_tokenize(text)

            tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

            return " ".join(tokens_without_sw)

        text = remove_stopwords(text)

        sentence = text

        return sentence

    def get_transcription(self, file_path=None):
        if not file_path:
            file_path = self.file_path

        speech, rate = librosa.load(file_path, sr=16000)

        input_values = self.wav2vec_tokenizer(speech, return_tensors='pt').input_values
        # Store logits (non-normalized predictions)
        with torch.no_grad():
            logits = self.wav2vec_model(input_values).logits

        # Store predicted id's
        predicted_ids = torch.argmax(logits, dim=-1)
        # decode the audio to generate text
        # Passing the prediction to the tokenzer decode to get the transcription
        return (self.wav2vec_tokenizer.batch_decode(predicted_ids)[0]).lower()

