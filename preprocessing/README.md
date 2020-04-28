## Steps to run preprocessing
* [Step 0] : pip install -r requirements.txt
* [Step 1] : Download lucene_index_9m.zip [here](https://drive.google.com/file/d/1ccZSys8u4F_mqNJ97OOlSLe3fjpFLhdv/view?usp=sharing) and extract it inside utils folder
* [Step 2] : Extract utils/all_movie_data.pkl.zip 
* [Step 3] : Extract output.zip
* [Step 4] : Run python preprocess.py batch_number batch_id (which takes the original set of 7680 movies and processes them in batches of size `batch_size`. It takes the batch_id as input as well which runs from 0 to floor(7680/batch_size)
* [Step 5] : Running the above step would create a json file in the output folder.
* [Step 6] : output/dataset_augmented_with_paraphrases_and_embedding_and_coref_and_skipthoughts.json contains the preprocessed ParaphraseRC dataset, i.e. for every movie, it contains a dictionary over each of its questions and its relevant subplot extracted by the preprocessing and the corresponding answer. It is organized in the form
```
{ 'movie_id': {
	'question1': {'relevant_plot': [Sentence_1, Sentence_2, ... Sentence_k], 'answers':[String_1,String_2, ... String_m]},
	'question2': {'relevant_plot': [Sentence_1, Sentence_2, ... Sentence_k], 'answers':[String_1,String_2, ... String_m]},
	   ...	
   }
}
```
* [Step 7] : Download SpanModel_BiDAF_data.zip [here](https://drive.google.com/file/d/1UsNderjmQ2xHCeS0GX3ff-eyud5s4T13/view?usp=sharing), which contains the preprocessed data in the BiDAF format
