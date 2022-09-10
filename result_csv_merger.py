from logging import exception
import os
import shutil
import time
from pydub import AudioSegment 
import shutil


class ResultsMerger:
    def __init__(self, input_path = None, output_path = None):

        if input_path:
            if not input_path.endswith('/'):
                input_path = input_path + '/'

        self.input_path = input_path

        if output_path:
            if not output_path.endswith('/'):
                output_path = output_path + '/'

        self.output_path = output_path
    
    def merge(self):
        # merge parts
        files = os.listdir(f"{self.input_path}parts/")
        for i, file in enumerate(files):
            try:
                shutil.move(f"{self.input_path}parts/{file}", f"{self.output_path}data/parts/")
                print(f"moving file {i} to parts")
            except Exception as e:
                if not file.endswith('csv'):
                    os.remove(f"{self.input_path}parts/{file}")
                print(f"erorr: {e}")
                print(f"removing {file}")



        # merge result
        files = os.listdir(f"{self.input_path}result/books to read")
        for i, file in enumerate(files):
            try:
                shutil.move(f"{self.input_path}result/books to read/{file}", f"{self.output_path}result/books to read/")
                print(f"moving file {i} to result")
            except Exception as e:
                if not file.endswith('csv'):
                    os.remove(f"{self.input_path}parts/{file}")
                print(f"erorr: {e}")
                print(f"removing {file}")

    def merge_csv(self):
        files = os.listdir(f"{self.input_path}parts/")
        files = [file for file in files if file.endswith('csv')]
        for i, file in enumerate(files):
            try:
                shutil.move(f"{self.input_path}parts/{file}", f"{self.output_path}data/parts/")
                print(f"moving file {i} to parts")
            except OSError as e:
                os.remove(f"{self.input_path}parts/{file}")
                print(f"removing {file}")
                print(f"erorr: {e}")


class FormatsHandler:
    '''
    read all the files, convert them to "wav", then remove them
    '''
    def __init__(self, directoryPath = 'drive1/new_audio_server/audio_search/data/'):
        self.directoryPath = directoryPath
        self.all_files = os.listdir(directoryPath)

    def handle_format(self, format = 'mp3'):
        # convert format to wav
        files = [file for file in self.all_files if file.endswith(format)]

        wav_files = [file for file in self.all_files if file.endswith('.wav')]

        # make wav files with same format to check if the file alaready converted before
        wav_files = [f[:-3] + format for f in wav_files]
        print(f"there are {len(files)} with {format} format")

        total_removed = 0
        for i, file in enumerate(files):
            if file in wav_files:
                print(f"skip {i}. {file}")
                continue
                
            try:
                print(f"converting {i}. {file} to wav \n")
                self.conver_format(file, format, saving_path = None)

                try:
                    #os.remove(os.path.join(self.directoryPath, file))
                    print(f"{i}. removing {file} \n")
                    total_removed += 1
                except:
                    print(f"error while removing      {os.path.join(self.directoryPath, file)}")
            except Exception as e:
            #     # write here all corrubpted files to texr
            #     # shutil.move(os.path.join(self.directoryPath, file), self.directoryPath + 'corrupted')
                print(f"error:   {e} ")

        print(f"totaaaaaaaal removed: {total_removed}")

    def conver_format(self, input_file_name, input_file_format, saving_path = None):
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

    def transform(self):
        print("+ "*30, "MP3", "+" *30)
        self.handle_format(format = 'mp3')

        print("+ "*30, "MP4", "+" *30)
        self.handle_format(format = 'mp4')

        print("+ "*30, "M4a", "+" *30)
        self.handle_format(format = 'm4a')








if __name__ == '__main__':

    while True:
        try:
            print("start Change formats")
            formathandler = FormatsHandler()
            formathandler.transform()
        except:
            pass


        try:
            print("start merging data0")
            merger = ResultsMerger(input_path = 'data0/', output_path = "drive1/new_audio_server/audio_search/")
            merger.merge_csv()
            print("start merging data1")
            merger = ResultsMerger(input_path = 'data1/', output_path = "drive1/new_audio_server/audio_search/")
            merger.merge_csv()
            print("start merging data2")
            merger = ResultsMerger(input_path = 'data2/', output_path = "drive1/new_audio_server/audio_search/")
            merger.merge_csv()
            time.sleep(10)
        except Exception as e:
            print(f"erorr: {e}")
