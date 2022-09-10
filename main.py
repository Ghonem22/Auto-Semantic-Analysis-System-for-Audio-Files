from audio_search import *
from semantic_search import *
from datetime import datetime
import time
from drive_cleaner import *


def main():

    wav2vec_model= Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_tokenizer =  Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    semantic_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    semantic_tokonizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


    i = 0
    while True:
        now = datetime.now()
        if now.second % 7 == 0:
            print(i)
            i += now.second
            time.sleep(.5)

        if now.minute % 3 == 0 :
            try:
                print("The audio search code is running now")
                spech_search_instance = spech_search_through_directory(directoryPath = 'drivex/new_audio_server/audio_search/data/', saving_path = 'drivex/new_audio_server/audio_search/', wav2vec_model = wav2vec_model, wav2vec_tokenizer = wav2vec_tokenizer, semantic_model = semantic_model, semantic_tokonizer = semantic_tokonizer, f = 12, i = 0)
                spech_search_instance.run_through_all_files(queries = ['read books'] )
                time.sleep(60) 
                    
            except:
                print("error while running audio search")
                time.sleep(60) 


        if False:
            try:
                print("The semantic search code is running now")
                search_multiple = semantic_search_through_directory(directoryPath = 'drivex/semantic_search/data', saving_path = "drivex/semantic_search/results")
                search_multiple.final_rocessing()
                time.sleep(60) 
            except:
                print("error while running semantic search")
                time.sleep(60) 

        if now.hour % 6  == 0 :
            try:
                print("drive cleaner running now")
                paths_to_clean = ['drivex/new_audio_server/audio_search/result/books to read',
                                'drivex/new_audio_server/audio_search/data/parts']
                config_path = 'drivex/new_audio_server/audio_search/data/configurations/config.txt'
                data_path = 'drivex/new_audio_server/audio_search/data'
                cleaner = driveCleaner(paths_to_clean = paths_to_clean,data_path = data_path, last_edit = 24, config_path = config_path, current_path = None,
                                extensions = ['wav'])

                cleaner.process()
                time.sleep(60) 
            except:
                print("error while running drivex cleaner")
                time.sleep(60) 



if __name__ == '__main__':
    main()
