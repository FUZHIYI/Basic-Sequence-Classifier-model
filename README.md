# Basic-Sequence-Classifier-model
single RNN, bidirectional RNN, HAN for sequence classification task on Imdb v1

## Introduction

Three sequence classifier model are implemented inside this repo:

1. single RNN
2. bidirectional RNN
   NOTE: support to using final_state  and averaged outputs fed into projection layer.  
3. HAN: hierarchical attention networks

It's based on the repo: https://github.com/triplemeng/hierarchical-attention-model. The code is for the Peking University Undergraduate Course "Web Data Mining". Dataset included are Imdb(v1) data (with only positive and negative labels) , and the homework sentence consistency data(not upload yet).



## How to use

1. download the Imdb(v1) data

   ```bash data.sh```
   
   It will download the raw Imdb data and uncompress it to ./data/aclImdb folder with positive samples under 'pos' and negative ones under 'neg' subdirectories.
   
2. pre-train word embeddings

       ```cd ./code
       python(3) gen_word_embeddings.py
       (By default, the embedding size is 50, it's usually used for Imdb data)```
       
   If you don't want to used pre-trained embedding, it could be configure later, but here it's needed for vocabulary statistics, too.
   
3. padding and truncating
   	If you want to train a single RNN or bidirectional RNN:
    
       python(3) preprocess_dataset.py --mode normal --max_sent_length 400
       
   the later one indicate truncating sequence length of dataset. It's equal to:
   
       python(3) preprocess_dataset.py -s 400
       
   You can choose how long your longest sentences are.
    	If you want to train a HAN (hierarchical attention model):
      
       python(3) preprocess_dataset.py --mode han
       
   It's equal to this:
   
       python(3) preprocess_dataset.py -m han --max_text_length 15 --max_sent_length 70
       
    Each text (review for Imdb data) will be composed of max_text_length sentences. If the original text is longer than that, we truncate it, and if shorter than that, we append empty sentences to it. And each sentence will be composed of max_sent_length words. If the original sentence is longer than that, we truncate it, and if shorter, we append the word of <pad> to it. Also, we keep track of the actual number of sentences each text contains, and actual number of words each sentence contains.	
    We directly read in pre-trained embeddings. Here we take the default dictionary size to be 10000. The words are indexed from 1 to 10000.Any words that are not included in the dictionary are marked as <unk>, and the index for <unk> is 0. The index for <pad> is 10001.
  
4. chose and configure your model
   open file train.py, find those codes around 40th line:
       
       #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
       #max_textlen, max_seqlen = df_train['text'][0].shape
       #model = HierarchicalAttentionClassifier(max_textlen=max_textlen, max_seqlen=max_seqlen, n_class=n_class, embed_matrix=embed_matrix, embed_trainable=False)
       #model = HierarchicalAttentionClassifier(max_textlen=max_textlen, max_seqlen=max_seqlen, n_class=n_class, vocab_size=10000, embed_size=50)
       
       (max_seqlen, ) = df_train['text'][0].shape
       #model = BiRNNSequenceClassifier(max_seqlen=max_seqlen, n_class=n_class, average_hidden=True, concat_fw_bw=True, embed_matrix=embed_matrix, embed_trainable=False)
       #model = BiRNNSequenceClassifier(max_seqlen=max_seqlen, n_class=n_class, average_hidden=False, concat_fw_bw=False, vocab_size=10000, embed_size=50)
       model = UniRNNSequenceClassifier(max_seqlen=max_seqlen, n_class=n_class, average_hidden=True, embed_matrix=embed_matrix, embed_trainable=False, lr=1e-4)
       #model = UniRNNSequenceClassifier(max_seqlen=max_seqlen, n_class=n_class, average_hidden=False, vocab_size=10000, embed_size=50)
       #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
       
   Uncomment corresponding model, set hyper-parameters you need.
   Note: if you want to use HAN model, comment `(max_seqlen, ) = df_train['text'][0].shape` and uncomment `#max_textlen, max_seqlen = df_train['text'][0].shape`.
   
5. run the model
   Train the model and evaluate it on the test set.
       
       python(3) train.py
       
   Hyper-parameters relative to training process are at the beginning lines around line 10.
       
       os.environ['CUDA_VISIBLE_DEVICES'] = '4'
       batch_size = 64
       n_class = 2
       n_epoch = 40
       output_dir = "../zzzz"


