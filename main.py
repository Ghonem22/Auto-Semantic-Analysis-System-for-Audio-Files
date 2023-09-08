from search.audio_search import *
from search.semantic_search import *

def main():

    wav2vec_model= Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_tokenizer =  Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    semantic_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    semantic_tokonizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    spech_search_instance = SpeechSearchThroughDirectory(directoryPath='drivex/new_audio_server/audio_search/data/',
                                                           saving_path='drivex/new_audio_server/audio_search/',
                                                           wav2vec_model=wav2vec_model,
                                                           wav2vec_tokenizer=wav2vec_tokenizer,
                                                           semantic_model=semantic_model,
                                                           semantic_tokonizer=semantic_tokonizer, f=12, i=0)
    spech_search_instance.run_through_all_files(queries=['read books'])



    search_multiple = SemanticSearchThroughDirectory(directoryPath='drivex/semantic_search/data',
                                                        saving_path="drivex/semantic_search/results")
    search_multiple.final_processing()



if __name__ == '__main__':
    main()
