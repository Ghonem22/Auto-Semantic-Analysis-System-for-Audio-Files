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


warnings.filterwarnings("ignore")

class Mixen:

    # Take dataframe and score's column name and return the groups with highest score 
    def get_segments(self, dfx, col):
        thresh = .65 * (max(dfx[col]) / 2)
        segments = []
        segment = []
        for index, row in dfx[dfx[col] > thresh].iterrows():
            
            if len(segment) == 0:
                segment.append(index)

            if index - segment[-1] < 4:
                segment.append(index)
            else:
                segment.sort()
                continus_segment = list(range(segment[0], segment[-1] +1 ))
                segments.append(continus_segment)
                segment = []
        segments = sorted(segments, key=len, reverse=True)
        return segments



    # take two time values and return their subtraction
    def subtract_time(self, val2, val1):
        val1 = list(map(float, val1.split(":")))
        val2 = list(map(float, val2.split(":")))

        if val2[-1] < val1[-1]:
            val2[-1] += 60
            val2[-2] -= 1

        if val2[-2] < val1[-2]:
            val2[-2] += 60
            val2[-3] -= 1

        return ":".join([str(round(val2[i] - val1[i], 3)) for i in range(len(val1))])

    # calculate the total Duration time for a group, to use that later to devide segments into < 60s segments
    def calc_time(self, segment, dfx):

        start = segment[0]
        end = segment[-1]
        
        Total_Duration = self.subtract_time(dfx.iloc[end]["Out"], dfx.iloc[start]["In"])
        time_list = list(map(float, Total_Duration.split(":")))
        time = time_list[0] * 60 * 60 + time_list[1] * 60 + time_list[2]
        return time

    # Devide groups into <60 s segments 
    new_segment = []
    def get_60s(self, segmentx, dfx):

        if len(segmentx) <= 1:
            new_segment.append(segmentx)
            return new_segment

        for j in range(len(segmentx)):
            time = self.calc_time(segmentx[0:j+1], dfx)
            # print(new_segment)

            if time > 60 and len(segmentx[0:j+1]) > 1:
                # print(time)
                # print(len(segment[j:]))

                seg1 = segmentx[0:j]
                seg2 = segmentx[j:]
                new_segment.append(seg1)
                # recursion
                self.get_60s(seg2, dfx)
                return new_segment

    # use all the groups indices, devide each one into < 60s segments
    def devide_long_segments_into_60s(self, segments, dfx):
        new_segments = []
        for i, segment in enumerate(segments):
            if self.calc_time(segment, dfx) <= 60 and self.calc_time(segment, dfx) >= 3:
                # print(f"segment {segment}")
                new_segments.append(segment)
            elif self.calc_time(segment, dfx) > 60:
                global new_segment
                new_segment = []
                new_segment = self.get_60s(segment, dfx)
                
                for i, seg in enumerate(new_segment):
                    if self.calc_time(seg, dfx) < 3:
                        del new_segment[i]
                        continue
                
                new_segments.extend(new_segment)
        return new_segments


    def Concat_time(self, dfx):
        # dfx.iloc[:]["In"] = dfx.iloc[0]["In"]
        # dfx.iloc[:]["Out"] = dfx.iloc[-1]["Out"]
        # dfx.iloc[:]["Duration"] = self.subtract_time(dfx.iloc[0]["Out"], dfx.iloc[0]["In"])
        dfx['Total Duration'] = self.subtract_time(dfx.iloc[-1]["Out"], dfx.iloc[0]["In"])

        return dfx 

    # Take dataframe and groups indices, devide the dataframe into that groups with defining the total duration time for that group
    def process_df(self, dfx,new_segments):
        # if len(new_segments) == 0:
        #     dfx['group'] = ''
        #     return dfx


        # print(f"new_segments: {new_segments}")
        # print(f"new_segments[0][0]: {new_segments[0][0]}")
        # print(f"new_segments[0][-1]: {new_segments[0][-1]}")
        # print(f"len(dfx):  {len(dfx)}")
        df_merged = dfx.iloc[new_segments[0][0]:new_segments[0][-1] +1]
        df_merged['group'] = '1'
        df_merged = self.Concat_time(df_merged)

        for i, segment in enumerate(new_segments):
            if i ==0:
                continue
            df1 = dfx.iloc[segment[0]:segment[-1] +1]

            df1['group'] = str(i+1)
            df1 = self.Concat_time(df1)
            df_merged = pd.concat([df_merged, df1], axis=0)
        return df_merged.set_index(['Total Duration', 'group'])




class semantic_search:
    def __init__(self, csvFilePath, model_ckpt = None, model = None, tokenizer = None):

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
        if not self._dataset:

            # make dataframe at huggingface dataset format
            print("reading and preparing dataset...")
            df = pd.read_csv(self.csvFilePath, sep = ',', header = 0, encoding= 'unicode_escape')
            self._dataset = Dataset.from_pandas(df)

        return self._dataset


    @property
    def model(self):
        # instantiate the model
        if not self._model:
            print("Loading Model...")
            self._model = AutoModel.from_pretrained(self.model_ckpt)
        return self._model

    @property
    def tokenizer(self):
        # instantiate the tokonizer
        if not self._tokenizer:
            print("Loading Tokonizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)

        return self._tokenizer

    @property
    def dataset_with_embeddings(self):
        # instantiate the tokonizer
        if not self._dataset_with_embeddings:
            print("Create new column with word embeding of the text...")

            self._dataset_with_embeddings = self.dataset.map(
                    lambda x: {"embeddings": self.get_embeddings(x["Text"]).cpu().numpy()[0]}
                )

        return self._dataset_with_embeddings.add_faiss_index(column="embeddings")


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


    def most_similart_sentence(self, query, n =3):

        try:
            query_embedding = self.get_embeddings([query]).cpu().detach().numpy()
            scores, most_similar = self.dataset_with_embeddings.get_nearest_examples(
                "embeddings", query_embedding, k=n
            )

            del most_similar["embeddings"]

            most_similar['score'] = scores

            most_similar = pd.DataFrame(most_similar)
            most_similar.sort_values(by=['score'], ascending=False, inplace= True)
            most_similar.reset_index(drop=True, inplace= True)

            return most_similar

        except KeyError:
            print("the csv file doesn't have Text column")
            return False
            
    def _dataset_conversion(self, hugging_dataset):
        '''
        Take huggingFace Dataset format and convert it into Dataframe
        '''
        data = {}
        for col in hugging_dataset.column_names:
            data[col] = hugging_dataset[col]
        
        return pd.DataFrame(data)

    def semantic_search(self, queries, sorting = None):

        # convert dataset into dataframe
        try:
            dataframe = self._dataset_conversion(self.dataset_with_embeddings)
            for query in queries:
                query_embedding = self.get_embeddings([query]).cpu().detach().numpy()

                dataframe[query] = dataframe.embeddings.apply(lambda x: round(float(util.pytorch_cos_sim(x, query_embedding)), 3))
            del dataframe['embeddings']

            if sorting:
                if sorting not in queries:
                    print("sorting should be one of the queries elements")
                
                else:
                    dataframe.sort_values(by=[sorting], ascending=False, inplace= True)
            return dataframe

        except KeyError:
            print("the csv file doesn't have Text column")
            return False



class semantic_search_through_directory(Mixen):
    def __init__(self, directoryPath = None, saving_path = None):

        if not saving_path:
            self.saving_path = os.getcwd()
        else:
            self.saving_path = saving_path

        if not directoryPath:
            directoryPath = os.getcwd()

        if not directoryPath.endswith('/'):
            directoryPath = directoryPath + '/'

        self.directoryPath =directoryPath
        # get all the csv file in the directory
        self.csv_files = glob.glob(self.directoryPath+'*.csv')
        self.processed_files_names = None
        # read names in /content/xxx/processed_files.txt
        # subtract them from self.csv_files

        # get the processed files names


    def most_similar_sentence_mutiple_files(self, queries, n = 3):

        for file in self.csv_files:
            print("\n\n\n********** Start processing {} **********\n".format(file.split("/")[-1]))
            search = semantic_search(csvFilePath = file )
        
            for query in queries:
                out = search.most_similart_sentence(query, n)
                try:
                    if out == None:
                        print("the csv file {}   doesn't have Text column".format(file.split("/")[-1]))

                except ValueError:
                    file_name = file.split("/")[-1].split(".")[0] + '_' + query + '.csv'
                    saving_name = os.path.join(self.saving_path, 'most_similar')
                    final_saving_name = os.path.join(saving_name, file_name)


                    isExist = os.path.exists(self.saving_path)

                    if not isExist:
                        # Create a new directory because it does not exist 
                        os.makedirs(self.saving_path)
                        print("Creating saving folder")

                    isExist2 = os.path.exists(saving_name)

                    if not isExist2:
                        # Create a new directory because it does not exist 
                        os.makedirs(saving_name)
                        print("Creating semantic_search folder")

                    out.to_csv(final_saving_name)

    def semantic_search_mutiple_files(self, queries, sorting = None):

        for file in self.csv_files:
            print("\n\n\n********** Start processing {} **********\n".format(file.split("/")[-1]))
            search = semantic_search(csvFilePath = file )
        
            out = search.semantic_search(queries= queries, sorting = sorting)
            try:
                if out == None:
                    print("the csv file {}   doesn't have Text column".format(file.split("/")[-1]))

            except ValueError:
                    file_name = file.split("/")[-1].split(".")[0] + '_' + 'semantic_search_result' + '.csv'
                    saving_name = os.path.join(self.saving_path, 'semantic_search')
                    final_saving_name = os.path.join(saving_name, file_name)

                    isExist = os.path.exists(self.saving_path)

                    if not isExist:
                        # Create a new directory because it does not exist 
                        os.makedirs(self.saving_path)
                        print("Creating saving folder")

                    isExist2 = os.path.exists(saving_name)

                    if not isExist2:
                        # Create a new directory because it does not exist 
                        os.makedirs(saving_name)
                        print("Creating semantic_search folder")

                    out.to_csv(final_saving_name)



    def final_rocessing(self, queries =  ['books', 'books to read', 'read', 'films', 'movies', 'movies to watch'], sorting = None):

        self.csv_files = glob.glob(self.directoryPath+'*.csv')

            
        # remove queries file and processed file if exist:
        try:
            os.remove(os.path.join(os.getcwd(), "queries.txt"))
        except:
            pass

        try:
            os.remove(os.path.join(os.getcwd(), "processed_files.txt"))
        except:
            pass



        # Get queries if exist
        try:
            shutil.copy(os.path.join(self.directoryPath, "queries.txt"), os.getcwd())
            queries_file = open(os.path.join(os.getcwd(), "queries.txt"),"r+") 
            queries_data = queries_file.readlines()
            queries_file.close() 

            queries = [query[:-1] for query in queries_data]
        except:
            pass

        print(f"queries:   {queries}")
        # Get processed files names if exist
        try:

            try:
                shutil.move(os.path.join(self.directoryPath, "processed_files.txt"), os.getcwd())
                

            except:
                os.remove(os.path.join(os.getcwd(), "processed_files.txt"))
                shutil.move(os.path.join(self.directoryPath, "processed_files.txt"), os.getcwd())
                

            else:
                pass

            processed_files = open(os.path.join(os.getcwd(), "processed_files.txt"),"r+") 
            processed_files_names = processed_files.readlines()
            processed_files.close() 
            self.processed_files_names = [file[:-1] for file in processed_files_names]
            
            print(f"processed_files is {self.processed_files_names}")

        except FileNotFoundError:
            self.processed_files_names = []
            print('processed_files.txt doesn\'t exist ')

        # remove any processed files
        self.csv_files = [file for file in self.csv_files if file.split("/")[-1] not in self.processed_files_names]
        print("file.split()[-1]:  {}".format([i.split("/")[-1] for i in self.csv_files]))
        print(f"self.csv_files:  {self.csv_files}")
        for file in self.csv_files:
            print("\n\n\n********** Start processing {} **********\n".format(file.split("/")[-1]))
            search = semantic_search(csvFilePath = file )

            for index, query in enumerate(queries):
                out = search.semantic_search(queries= [query], sorting = None)

                try:
                    if out == None:
                        print("the csv file {}   doesn't have Text column".format(file.split("/")[-1]))

                except ValueError:
                        segments = self.get_segments(out, col = query)
                        new_segments = self.devide_long_segments_into_60s(segments, out)
                        if len(new_segments) == 0:
                            result = search.semantic_search(queries= [query], sorting = query)
                            result = result[result[query] > .4 * max(result[query])]
                        else:
                            result = self.process_df(out,new_segments)

                        file_name = file.split("/")[-1]
                        saving_name = os.path.join(self.saving_path, query)
                        final_saving_name = os.path.join(saving_name, file_name)

                        isExist = os.path.exists(self.saving_path)

                        if not isExist:
                            # Create a new directory because it does not exist 
                            os.makedirs(self.saving_path)
                            print("Creating saving folder")

                        isExist2 = os.path.exists(saving_name)

                        if not isExist2:
                            # Create a new directory because it does not exist 
                            os.makedirs(saving_name)
                            print(f"Creating folder for {query}")

                        # if processed_files.txt not exist, creat it
                        # append file_name in new line

                        result.to_csv(final_saving_name)
                        # save the file name just once
                        if index == 0 and file_name not in self.processed_files_names:
                            processed_files = open(os.path.join(os.getcwd(), "processed_files.txt"),"a+")
                            processed_files.writelines(file_name +  '\n')
                            processed_files.close() 
                            
                except:
                    print(" error while processing the file, please chech the file structure")
        try:
            shutil.move(os.path.join(os.getcwd(), "processed_files.txt"),self.directoryPath)
        except:
            pass

        try:
            os.remove(os.path.join(os.getcwd(), "queries.txt"))
        except:
            pass
			