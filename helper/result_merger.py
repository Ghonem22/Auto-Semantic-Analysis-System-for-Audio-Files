from logging import exception
import os
import shutil
import time
from pydub import AudioSegment 


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
        # files = os.listdir(f"{self.input_path}parts/")
        # for i, file in enumerate(files):
        #     try:
        #         shutil.move(f"{self.input_path}parts/{file}", f"{self.output_path}data/parts/")
        #         print(f"moving file {i} to parts")
        #     except Exception as e:
        #         if not file.endswith('csv'):
        #             os.remove(f"{self.input_path}parts/{file}")
        #         print(f"erorr: {e}")
        #         print(f"removing {file}")



        # merge result
        files = os.listdir(f"{self.input_path}result/books to read")
        for i, file in enumerate(files):
            try:
                shutil.move(f"{self.input_path}result/books to read/{file}", f"{self.output_path}result/books to read/")
                print(f"moving file {i} to result")
            except Exception as e:
                os.remove(f"{self.input_path}result/books to read/{file}")
                print(f"erorr: {e}")
                print(f"removing {file}")

    def merge_csv(self):
        files = os.listdir(f"{self.input_path}parts/")
        files = [file for file in files if file.endswith('csv')]
        for i, file in enumerate(files):
            try:
                shutil.move(f"{self.input_path}parts/{file}", f"{self.output_path}data/parts/")
                print(f"moving file {i} to parts")
            except Exception as e:
                if not file.endswith('csv'):
                    os.remove(f"{self.input_path}parts/{file}")
                    print(f"removing {file}")
                print(f"erorr: {e}")





if __name__ == '__main__':
    while True:
        try:
            print("start merging data0")
            merger = ResultsMerger(input_path = 'data0/', output_path = "drive1/new_audio_server/audio_search/")
            merger.merge_csv()
            merger.merge()
            print("start merging data1")
            merger = ResultsMerger(input_path = 'data1/', output_path = "drive1/new_audio_server/audio_search/")
            merger.merge_csv()
            merger.merge()
            print("start merging data2")
            merger = ResultsMerger(input_path = 'data2/', output_path = "drive1/new_audio_server/audio_search/")
            merger.merge_csv()
            merger.merge()
            time.sleep(10)
        except Exception as e:
            print(f"erorr: {e}")

