import numpy as np
import pickle as pkl
import pandas as pd
import os, nltk, argparse, json
from gensim.models import Word2Vec
from tensorflow.contrib.keras import preprocessing

# Get embed_matrix(np.ndarray), word2index(dict) and index2word(dict). All of them including extra unknown word "<unk>" and padding word "<pad>", that is, returned size = param size + 2.
#   embedding_model: A pre-trained gensim.models.Word2Vec model.
def build_emb_matrix_and_vocab(embedding_model, keep_in_dict=10000, embedding_size=50):
    # 0 th element is the default one for unknown words, and keep_in_dict+1 th element is used as padding.
    emb_matrix = np.zeros((keep_in_dict+2, embedding_size))
    word2index = {}
    index2word = {}
    for k in range(1, keep_in_dict+1):
        word = embedding_model.wv.index2word[k-1]
        word2index[word] = k
        index2word[k] = word
        emb_matrix[k] = embedding_model[word]
    word2index['<unk>'] = 0
    index2word[0] = '<unk>'
    word2index['<pad>'] = keep_in_dict+1 
    index2word[keep_in_dict+1] = '<pad>'
    return emb_matrix, word2index, index2word

# Get an sentence (list of words) as list of index. All words change into lower form.
def __sent2index(wordlist, word2index):
    wordlist = [word.lower() for word in wordlist]
    sent_index = [word2index[word] if word in word2index else 0 for word in wordlist]
    return sent_index

# Read data from directory <data_dir>, return a list (text) of list (sent) of list (word index).
def __gen_data_imdbv1(data_dir, word2index, forHAN):
    data = []
    for filename in os.listdir(data_dir):
        file = os.path.join(data_dir, filename)
        with open(file) as f:
            content = f.readline()
            if forHAN:
                sent_list = nltk.sent_tokenize(content)
                sents_word = [nltk.word_tokenize(sent) for sent in sent_list]
                sents_index = [__sent2index(wordlist, word2index) for wordlist in sents_word]
                data.append(sents_index)
            else:
                word_list = nltk.word_tokenize(content)
                words_index = __sent2index(word_list, word2index)
                data.append(words_index)
    return data

# Read data from directory <data_dir>, return a list (text) of list (sent) of list (word index).
def __gen_data_scdata(data_file, word2index, forHAN, for_infer=False):
    data = []
    label = []
    with open(data_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            jsob = json.loads(line)
            if not for_infer:
                label.append(int(jsob['label']))
            content = jsob['text']
            if forHAN:
                sent_list = nltk.sent_tokenize(content)
                sents_word = [nltk.word_tokenize(sent) for sent in sent_list]
                sents_index = [__sent2index(wordlist, word2index) for wordlist in sents_word]
                data.append(sents_index)
            else:
                word_list = nltk.word_tokenize(content)
                words_index = __sent2index(word_list, word2index)
                data.append(words_index)
    return data, label

# Pass in indexed dataset, padding and truncating to corresponding length in both text & sent level.
#   return data_formatted(after padding&truncating), text_lens(number of sents), text_sent_lens(number of words in each sents inside the text)
def preprocess_text_HAN(data, max_sent_len, max_text_len, keep_in_dict=10000):
    text_lens = [] # how many sents in each text
    text_sent_lens = [] # a list of list, how many words in each no-padding sent 
    data_formatted = [] # padded and truncated data

    for text in data:
        
        # 1. text_lens
        sent_lens = [len(sent) for sent in text]
        text_len = len(sent_lens)
        text_right_len = min(text_len, max_text_len)
        text_lens.append(text_right_len)

        # 2. text_sent_lens & data_formatted
        sent_right_lens = [min(sent_len, max_sent_len) for sent_len in sent_lens]
        text_formatted = preprocessing.sequence.pad_sequences(text, maxlen=max_sent_len, padding="post", truncating="post", value=keep_in_dict+1)
        
        # sent level's padding & truncating are both done, here are padding and truncating in text level below.
        lack_text_len = max_text_len - text_len        
        if lack_text_len > 0: 
            # padding
            sent_right_lens += [0]*lack_text_len
            extra_rows = np.full((lack_text_len, max_sent_len), keep_in_dict+1) # complete-paddinged sents
            text_formatted_right_len = np.append(text_formatted, extra_rows, axis=0)
        elif lack_text_len < 0:
            # truncating
            sent_right_lens = sent_right_lens[:max_text_len]
            row_index = [max_text_len+i for i in list(range(0, -lack_text_len))]
            text_formatted_right_len = np.delete(text_formatted, row_index, axis=0)
        else: 
            # exactly, nothing to do
            text_formatted_right_len = text_formatted

        text_sent_lens.append(sent_right_lens)
        data_formatted.append(text_formatted_right_len)

    return data_formatted, text_lens, text_sent_lens

# Pass in indexed dataset, padding and truncating to corresponding length in sent level.
#   return data_formatted(after padding&truncating), sent_lens(number of words inside the sent)
def preprocess_text(data, max_sent_len, keep_in_dict=10000):

    # 1. sent_lens
    sent_lens = []
    for sent in data:
        sent_len = len(sent)
        sent_right_len = min(sent_len, max_sent_len)
        sent_lens.append(sent_right_len)
        
    #2. data_formatted
    data_formatted = preprocessing.sequence.pad_sequences(data, maxlen=max_sent_len, padding="post", truncating="post", value=keep_in_dict+1)
    #print(type(data_formatted))
    data_formatted = list(data_formatted)

    return data_formatted, sent_lens


# do all things above and save.
def imdbv1(working_dir="../data/aclImdb", forHAN=False):

    #============================================================
    # 1. embedding matrix, word2index table, index2word table
    #============================================================
    fname = os.path.join(working_dir, "imdb_embedding")
    if os.path.isfile(fname):
        embedding_model = Word2Vec.load(fname)
    else:
        print("please run gen_word_embeddings.py first to generate embeddings!")
        exit(1)
    print("generate word2index and index2word, get corresponding-sized embedding maxtrix...")
    emb_matrix, word2index, index2word = build_emb_matrix_and_vocab(embedding_model)

    #================================================================
    # 2. indexed dataset: number/int representation, not string
    #================================================================
    print("tokenizing and word-index-representing...")

    train_dir = os.path.join(working_dir, "train")
    train_pos_dir = os.path.join(train_dir, "pos")
    train_neg_dir = os.path.join(train_dir, "neg")

    test_dir = os.path.join(working_dir, "test")
    test_pos_dir = os.path.join(test_dir, "pos")
    test_neg_dir = os.path.join(test_dir, "neg")

    train_pos_data = __gen_data_imdbv1(train_pos_dir, word2index, forHAN)
    train_neg_data = __gen_data_imdbv1(train_neg_dir, word2index, forHAN)
    train_data = train_neg_data + train_pos_data

    test_pos_data = __gen_data_imdbv1(test_pos_dir, word2index, forHAN)
    test_neg_data = __gen_data_imdbv1(test_neg_dir, word2index, forHAN)
    test_data = test_neg_data + test_pos_data

    #================================
    # 3. padding and truncating
    #================================
    print("padding and truncating...")
    if forHAN:
        x_train, train_text_lens, train_text_sent_lens = preprocess_text_HAN(train_data, max_sent_length, max_text_length)
        x_test, test_text_lens, test_text_sent_lens = preprocess_text_HAN(test_data, max_sent_length, max_text_length)
    else:
        x_train, train_sent_lens = preprocess_text(train_data, max_sent_length)
        x_test, test_sent_lens = preprocess_text(test_data, max_sent_length)
    y_train = [0]*len(train_neg_data)+[1]*len(train_pos_data)
    y_test = [0]*len(test_neg_data)+[1]*len(test_pos_data)

    #===============
    # 4. saving
    #===============
    print("save word embedding matrix...")
    emb_filename = os.path.join(working_dir, "emb_matrix")
    pkl.dump([emb_matrix, word2index, index2word], open(emb_filename, "wb")) 

    print("save data for training...")
    if forHAN:
        df_train = pd.DataFrame({'text':x_train, 'label':y_train, 'text_length':train_text_lens, 'sents_length':train_text_sent_lens})
    else:
        df_train = pd.DataFrame({'text':x_train, 'label':y_train, 'text_length':train_sent_lens})
    train_filename = os.path.join(working_dir, "train_df_file")
    df_train.to_pickle(train_filename)

    print("save data for testing...")
    if mode=="han":
        df_test = pd.DataFrame({'text':x_test, 'label':y_test, 'text_length':test_text_lens, 'sents_length':test_text_sent_lens})
    else:
        df_test = pd.DataFrame({'text':x_test, 'label':y_test, 'text_length':test_sent_lens})
    test_filename = os.path.join(working_dir, "test_df_file")
    df_test.to_pickle(test_filename)

# do all things above and save.
def scdata(working_dir="../data/sentence_consistency_data", forHAN=False):

    #============================================================
    # 1. embedding matrix, word2index table, index2word table
    #============================================================
    fname = os.path.join(working_dir, "scdata_embedding")
    if os.path.isfile(fname):
        embedding_model = Word2Vec.load(fname)
    else:
        print("please run gen_word_embeddings.py first to generate embeddings!")
        exit(1)
    print("generate word2index and index2word, get corresponding-sized embedding maxtrix...")
    emb_matrix, word2index, index2word = build_emb_matrix_and_vocab(embedding_model)

    #================================================================
    # 2. indexed dataset: number/int representation, not string
    #================================================================
    print("tokenizing and word-index-representing...")

    train_file = os.path.join(working_dir, "train_data")
    test_file = os.path.join(working_dir, "test_data")
    valid_file = os.path.join(working_dir, "valid_data")

    train_data, y_train = __gen_data_scdata(train_file, word2index, forHAN)
    test_data, _ = __gen_data_scdata(test_file, word2index, forHAN, for_infer=True)
    valid_data, y_valid = __gen_data_scdata(valid_file, word2index, forHAN)

    #================================
    # 3. padding and truncating
    #================================
    print("padding and truncating...")
    if forHAN:
        x_train, train_text_lens, train_text_sent_lens = preprocess_text_HAN(train_data, max_sent_length, max_text_length)
        x_test, test_text_lens, test_text_sent_lens = preprocess_text_HAN(test_data, max_sent_length, max_text_length)
        x_valid, valid_text_lens, valid_text_sent_lens = preprocess_text_HAN(valid_data, max_sent_length, max_text_length)
    else:
        x_train, train_sent_lens = preprocess_text(train_data, max_sent_length)
        x_test, test_sent_lens = preprocess_text(test_data, max_sent_length)
        x_valid, valid_sent_lens = preprocess_text(valid_data, max_sent_length)

    #===============
    # 4. saving
    #===============
    print("save word embedding matrix...")
    emb_filename = os.path.join(working_dir, "emb_matrix")
    pkl.dump([emb_matrix, word2index, index2word], open(emb_filename, "wb")) 

    print("save data for training...")
    if forHAN:
        df_train = pd.DataFrame({'text':x_train, 'label':y_train, 'text_length':train_text_lens, 'sents_length':train_text_sent_lens})
    else:
        df_train = pd.DataFrame({'text':x_train, 'label':y_train, 'text_length':train_sent_lens})
    train_filename = os.path.join(working_dir, "train_df_file")
    df_train.to_pickle(train_filename)

    print("save data for testing...")
    if forHAN:
        df_test = pd.DataFrame({'text':x_test, 'text_length':test_text_lens, 'sents_length':test_text_sent_lens})
    else:
        df_test = pd.DataFrame({'text':x_test, 'text_length':test_sent_lens})
    test_filename = os.path.join(working_dir, "test_df_file")
    df_test.to_pickle(test_filename)

    print("save data for validating...")
    if mode=="han":
        df_valid = pd.DataFrame({'text':x_valid, 'label':y_valid, 'text_length':valid_text_lens, 'sents_length':valid_text_sent_lens})
    else:
        df_valid = pd.DataFrame({'text':x_valid, 'label':y_valid, 'text_length':valid_sent_lens})
    valid_filename = os.path.join(working_dir, "valid_df_file")
    df_valid.to_pickle(valid_filename)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some important parameters.')
    parser.add_argument('-m', '--mode', type=str, default='normal', help='generate dataset for HAN or normal sequence model.')
    parser.add_argument('-dn', '--dataname', type=str, default='imdbv1', help='imdbv1 or scdata.')
    parser.add_argument('-s', '--max_sent_length', type=int, default=70, help='fix the sentence length in all texts.')
    parser.add_argument('-t', '--max_text_length', type=int, default=15, help='fix the maximum text length in terms of senence.')

    args = parser.parse_args()
    mode = args.mode.lower()
    dataname = args.dataname.lower()
    max_sent_length = args.max_sent_length
    max_text_length = args.max_text_length
    assert mode in ["han", "normal"], "unknown data preprocessing mode."
    assert dataname in ["imdbv1", "scdata"], "unknown dataset."

    print("data preprocessing mode is {}".format(mode.upper()))
    print("dataset pre-processed is {}".format(dataname))
    print('max sent length is set as {}'.format(max_sent_length))
    forHAN = mode=="han"
    if forHAN:
        print('max text length is set as {}'.format(max_text_length))

    if dataname=="imdbv1":
        imdbv1(forHAN=forHAN)
    else:
        scdata(forHAN=forHAN)



