# Basic-Sequence-Classifier-model
Single RNN, bidirectional RNN, HAN for sequence classification task on Imdb v1 & a sentence consistency dataset.

## Introduction

Three sequence classifier model are implemented inside this repo:

1. single RNN
2. bidirectional RNN
   NOTE: support to using final state and averaging hidden states as input of projection layer. 
3. HAN: hierarchical attention networks

It's based on the repo: https://github.com/triplemeng/hierarchical-attention-model. This repo is for project of Peking University undergraduate course "Web Data Mining". Dataset includes Imdb version 1  (with only positive and negative labels) , and the homework sentence consistency dataset.



## How to use

1. download the Imdb(v1) data

   ```
   bash data.sh
   ```

   It will download the raw Imdb data and uncompress it to ./data/aclImdb folder with positive samples under 'pos' and negative ones under 'neg' subdirectories.

2. pre-train word embeddings

    ```
    cd ./code
    python gen_word_embeddings.py
    #By default, the embedding size is 50, it's usually used for Imdb data)
    ```

  If you don't want to used pre-trained embedding, it could be configured later, but here it's still needed for vocabulary statistics.

3. padding and truncating
    If you want to train a single RNN or bidirectional RNN:

    ```
    python preprocess_dataset.py --mode normal --max_sent_length 400
    ```
    the later param indicates truncating length of seqs. It's equal to:

    ```
    python preprocess_dataset.py -s 400
    ```

    You can choose how long your longest sentences are. If you want to train a HAN (hierarchical attention model):

    ```
    python preprocess_dataset.py --mode han
    ```

    It's equal to this:

    ```
    python preprocess_dataset.py -m han --max_text_length 15 --max_sent_length 70
    ```

    Each text (review for Imdb data) will be composed of max_text_length sentences. If the original text is longer than that, we truncate it, and if shorter than that, we append empty sentences to it. And each sentence will be composed of max_sent_length words. If the original sentence is longer than that, we truncate it, and if shorter, we append the replaced word `<pad>` to it. Also, we keep track of the actual number of sentences each text contains, and actual number of words each sentence contains.	
    We directly read in pre-trained embeddings. Here we take the default dictionary size to be 10000. The words are indexed from 1 to 10000. Any words that are not included in the dictionary are marked as `<unk>`, and the index for `<unk>` is 0. The index for `<pad>` is 10001.


4. run the model

   ```
   python train.py
   ```



