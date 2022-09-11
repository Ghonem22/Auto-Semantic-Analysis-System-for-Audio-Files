import pandas as pd
import os
import warnings
import shutil
from pydub import AudioSegment
from pydub.utils import make_chunks
import nltk
import json

warnings.filterwarnings("ignore")
nltk.download('stopwords')
nltk.download('punkt')

from utilities.helper import *
from utilities.s3_utill import *


class spech_search_through_directory:

    def __init__(self, audio_path=None, directoryPath=None, saving_path=None, current_path=None, wav2vec_model=None,
                 wav2vec_tokenizer=None, semantic_model=None, semantic_tokonizer=None, f=10, i=0):

        # create result folder and save result audio in it
        if not saving_path:
            self.saving_path = os.getcwd()
        else:
            self.saving_path = saving_path

        # path of data/ config
        if not directoryPath:
            directoryPath = os.getcwd()

        if directoryPath:
            if not directoryPath.endswith('/'):
                directoryPath = directoryPath + '/'

        # path of the server
        if current_path:
            if not current_path.endswith('/'):
                current_path = current_path + '/'

        # path of the server
        if audio_path:
            if not audio_path.endswith('/'):
                audio_path = audio_path + '/'

        self.audio_path = audio_path
        self.directoryPath = directoryPath
        self.corrubted = []
        # get all the csv file in the directory
        self.processed_audios_names = []

        self.wav2vec_model = wav2vec_model
        self.wav2vec_tokenizer = wav2vec_tokenizer
        self.semantic_model = semantic_model
        self.semantic_tokonizer = semantic_tokonizer
        # read names in /content/xxx/processed_files.txt
        # subtract them from self.csv_files

        if not current_path:
            self.current_path = os.getcwd()
        else:
            self.current_path = current_path
        print(f"current_path: {self.current_path} \n")

        # get the  files names

        all_files = os.listdir(audio_path)
        original_wav_files = [file for file in all_files if file.endswith('.wav')]
        print(len(original_wav_files))

        # for file in all_wav_files[:30]:
        #     shutil.move(os.path.join(directoryPath, file), 'data/')

        # self.all_wav_files = [f for f in os.listdir('data/') if f.endswith('wav')]

        # get processed files from parts folder
        processed_audios_names = os.listdir(audio_path + 'parts')
        processed_audios_names = [f.replace('_scores', '').replace('csv', 'wav') for f in processed_audios_names if
                                  f.endswith('csv')]

        # get processed files from txt file
        processed_audios_names2 = []
        try:
            shutil.copy(os.path.join(audio_path, "configurations/processed_audio_files.txt"), self.current_path)

            processed_files = open(os.path.join(self.current_path, "processed_audio_files.txt"), "r+")
            processed_audios_names2 = processed_files.readlines()
            processed_files.close()
            processed_audios_names2 = [file[:-1] for file in processed_audios_names2]
            print(f"processed from text file: {len(processed_audios_names2)}")
            os.remove(os.path.join(self.current_path, "processed_audio_files.txt"))

        except:
            pass

        processed_audios_names.extend(processed_audios_names2)

        print(f"processed from parts: {len(processed_audios_names) - len(processed_audios_names2)}")

        original_wav_files = [file for file in original_wav_files if file not in processed_audios_names]
        original_wav_files.sort()

        # Here we want to divide data into segments so each process process segment
        n1 = int((len(original_wav_files) / f)) * i
        n2 = int((len(original_wav_files) / f)) * (i + 1)
        print(f"n1111111111: {n1}\n n2222222222222: {n2}")

        num = 10

        if n2 > n1 + num:
            n2 = n1 + num

        print(f"new n222222222222: {n2}")
        original_wav_files = original_wav_files[n1:n2]
        print("+" * 50)
        print(f"all_wav_files:  {original_wav_files}")
        print("+" * 50)

        all_wav_files = [file for file in os.listdir(self.directoryPath) if file.endswith('.wav')]
        if len(all_wav_files) <= num:

            for i, file in enumerate(original_wav_files):
                try:
                    shutil.copy(f"{audio_path}{file}", self.directoryPath)
                    print(f"moveing {i}. {file} to data")
                except Exception as e:
                    print(f"error while moving file to data: {e}")

        all_wav_files = [file for file in os.listdir(self.directoryPath) if file.endswith('.wav')]

        # get processed files from parts folder
        processed_audios_names3 = os.listdir(self.directoryPath + 'parts')
        processed_audios_names3 = [f.replace('_scores', '').replace('csv', 'wav') for f in processed_audios_names3 if
                                   f.endswith('csv')]

        processed_audios_names.extend(processed_audios_names3)

        # remove processed files
        for file in all_wav_files:
            if file in processed_audios_names:
                try:
                    os.remove(self.directoryPath + file)
                    print(f"removing {self.directoryPath}{file}")
                except:
                    print(f"can't remove {file} from {self.directoryPath}")

        self.all_wav_files = [file for file in all_wav_files if file not in processed_audios_names]

        print(f"new {len(self.all_wav_files)}  files: \n {self.all_wav_files}")

    def conver_format(self, input_file_name, input_file_format, saving_path=None):
        input_file_path = os.path.join(self.directoryPath, input_file_name)
        m4a_audio = AudioSegment.from_file(input_file_path, format=input_file_format)
        wav_name = input_file_name.split("/")[-1].split(".")[0] + ".wav"

        if not saving_path:
            saving_path = self.directoryPath

        saving_wav_name = os.path.join(saving_path, wav_name)

        wav_Exist = os.path.exists(saving_wav_name)
        if not wav_Exist:
            m4a_audio.export(saving_wav_name, format="wav")
        else:
            print("wav file alerady exist \n")

    def get_sgements_with_high_score(self, file_name, folder_path=None, saving_folder=None, queries=['books to read'],
                                     thresh_score=.06, audio_length=30):

        if not folder_path:
            folder_path = self.directoryPath

        if not saving_folder:
            saving_folder = self.saving_path

        seconds = audio_length

        file_with_path = os.path.join(folder_path, file_name)
        print(file_with_path)
        myaudio = AudioSegment.from_file(file_with_path, "wav")
        chunk_length_ms = seconds * 1000  # pydub calculates in millisec
        chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of n sec

        try:
            os.makedirs(saving_folder)
        except:
            pass

        try:
            os.makedirs(os.path.join(folder_path, 'parts'))  # creating a folder named parts
        except:
            pass

        try:
            os.makedirs(os.path.join(self.current_path, 'parts'))  # creating a folder named parts
        except:
            pass

        try:
            os.makedirs(os.path.join(saving_folder,
                                     'result'))  # creating a folder named resutls for saving .wav files with acceptable score
        except:
            pass

        chunks_scores = {"chunks_names": []}
        for i, chunk in enumerate(chunks):
            chunk_Exist = None
            chunk_full_name = file_name.split(".")[0] + f"_{i:03}.wav"

            # to save inside folder with file name inside parts folder
            # chunck_saving = os.path.join(os.path.join(folder_path, 'parts/'), file_name.split(".")[0] + '/')
            chunck_saving = os.path.join(folder_path, 'parts/')

            try:
                os.makedirs(chunck_saving)  # creating a folder chunck_saving parts
            except:
                pass

            chunk_name = chunck_saving + chunk_full_name

            chunk_name_for_analysis = os.path.join(self.current_path, 'parts/') + chunk_full_name
            print("exporting", chunk_full_name)

            # save chunk in parts folder in directorypath
            # try:
            #     chunk.export(chunk_name, format="wav")
            # except:
            #     print("chunk_name")
            try:
                chunk_Exist = os.path.exists(chunk_name_for_analysis)
                if chunk_Exist:
                    continue
                chunk.export(chunk_name_for_analysis, format="wav")
            except:
                print("chunk_name_for_analysis")

            if chunk_Exist:
                continue

            search = spech_search(file_path=chunk_name_for_analysis, wav2vec_model=self.wav2vec_model,
                                  wav2vec_tokenizer=self.wav2vec_tokenizer)
            transcription = search.get_transcription()
            semantic_search_via_audio = semantic_search_for_audio(transcription, model=self.semantic_model,
                                                                  tokenizer=self.semantic_tokonizer)
            os.remove(chunk_name_for_analysis)

            chunks_scores["chunks_names"].append(chunk_full_name)
            for query in queries:

                # create folder for query if not exist
                saving_folder_for_queries = os.path.join(saving_folder, 'result')

                saving_query = os.path.join(saving_folder_for_queries, query)
                try:
                    # Create a new directory because it does not exist 
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

        print(f"chunks_scores_saving_name:   {chunks_scores_saving_name}")
        is_uploaded = save_csv(chunks_scores_dataframe, chunks_scores_saving_name, '/data/parts/')
        # chunks_scores_dataframe.to_csv(chunks_scores_saving_name, index=True)

    def run_through_all_files(self, running_path=None, queries=['read books', 'watch movies']):

        running_path = self.current_path

        # remove queries file and processed file if exist:
        try:
            os.remove(os.path.join(os.getcwd(), "config.txt"))
        except:
            pass

        # get processed files names from parts folder

        # try:
        #     try:
        #         # move processed_audio_files
        #         shutil.move(os.path.join(self.directoryPath, "configurations/processed_audio_files.txt"), running_path)
        #         print("processed_audio_files.txt moved from directory path to running path \n")
        #         processed_files = open(os.path.join(running_path, "processed_audio_files.txt"),"r+") 
        #         processed_audios_names = processed_files.readlines()
        #         processed_files.close() 
        #         self.processed_audios_names = [file[:-1] for file in processed_audios_names]

        #         print(f"len processed_audio_files is: {len(self.processed_audios_names)}  \n")

        #     except:

        #         print(" processed_audio_files.txt not in drive, reading it  from server \n")
        #         processed_files = open(os.path.join(running_path, "processed_audio_files.txt"),"r+") 
        #         processed_audios_names = processed_files.readlines()
        #         processed_files.close() 
        #         self.processed_audios_names = [file[:-1] for file in processed_audios_names]

        #         print(f"len processed_audio_files is: {len(self.processed_audios_names)}  \n")

        # except:
        #     print("processed_audio_files.txt doesn't exist")

        # read processed files

        # Get queries if exist

        try:
            shutil.copy(os.path.join(self.directoryPath, "configurations/config.txt"), running_path)

            config = json.load(
                open(os.path.join(running_path, "config.txt")))  # audio_legnth  queries    score_threshold

            queries = config['queries']
            thresh_score = float(config['score_threshold'])
            audio_length = config['audio_length']
            print(f"get quiries succcccccc :\n\n\n       {queries} \n {thresh_score}  \n {audio_length}")
        except:
            queries = ["books to read", "movies to watch"]
            thresh_score = 0.11
            audio_length = 30
            print(f"get quiries wrrrrrong :\n       {queries} \n {thresh_score}  \n {audio_length}")

        thresh_score -= .05

        for each_file in self.all_wav_files:
            # try:
            #     # check if any file is corrubted
            #     MP3(os.path.join(self.directoryPath, each_file))
            self.get_sgements_with_high_score(each_file, self.directoryPath, self.saving_path, queries=queries,
                                              thresh_score=thresh_score, audio_length=audio_length)
            # processed_files = open(os.path.join(running_path, "processed_audio_files.txt"),"a+")
            # processed_files.writelines(each_file +  '\n')
            # processed_files.close()

            # remove files after processing
            try:
                os.remove(os.path.join(self.directoryPath, each_file))
                print(f"removing {each_file} after processing it")
            except:
                print(f" can't remove {each_file} after processing ")

            # except:
            #     print(f"file {os.path.join(self.directoryPath, each_file)} is corrupted")
            #     shutil.move(os.path.join(self.directoryPath, each_file), self.directoryPath + 'corrupted')

        # try:
        #     os.remove(os.path.join(os.path.join(self.directoryPath, "configurations"), "processed_audio_files.txt"))
        # except:
        #     pass

        # # move the processed_files.txt again to the files directory
        # try:
        #     shutil.move(os.path.join(running_path, "processed_audio_files.txt"), os.path.join(self.directoryPath, "configurations"))
        # except:
        #     pass

        # remove queries file from the working directory
        try:
            os.remove(os.path.join(os.getcwd(), "config.txt"))
        except:
            pass

    def after_process(self):
        for i, file in enumerate(self.all_wav_files):
            os.remove(f"{self.directoryPath}{file}")
            print(f"reveing {i}. {file} from data")
