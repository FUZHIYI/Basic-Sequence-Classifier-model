import os, nltk, json, argparse
from gensim.models import Word2Vec

def gen_formatted_data_imdbv1(data_dir):
    data = []
    for filename in os.listdir(data_dir):
        file = os.path.join(data_dir, filename)
        with open(file, 'r') as f:
            content = f.readline().lower()
            content_formatted = nltk.word_tokenize(content)
            data.append(content_formatted)            
    return data

def gen_embedding_model_imdbv1(working_dir='../data/aclImdb', embedding_size=50):

    fname = os.path.join(working_dir, "imdb_embedding")
    if os.path.isfile(fname):
        embedding_model = Word2Vec.load(fname)
        return embedding_model

    train_dir = os.path.join(working_dir, "train")
    train_pos_dir = os.path.join(train_dir, "pos")
    train_neg_dir = os.path.join(train_dir, "neg")

    test_dir = os.path.join(working_dir, "test")
    test_pos_dir = os.path.join(test_dir, "pos")
    test_neg_dir = os.path.join(test_dir, "neg")
    
    train = gen_formatted_data_imdbv1(train_pos_dir) + gen_formatted_data_imdbv1(train_neg_dir)
    test = gen_formatted_data_imdbv1(test_pos_dir) + gen_formatted_data_imdbv1(test_neg_dir)
    alldata = train + test

    embedding_model = Word2Vec(train, size=embedding_size, window=5, min_count=5)
    embedding_model.save(fname)

    return embedding_model

def gen_formatted_data_scdata(data_file):
    data = []
    with open(data_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            new_content = json.loads(line)
            text = nltk.word_tokenize(new_content['text'])
            data.append(text)
    return data

def gen_embedding_model_scdata(working_dir='../data/sentence_consistency_data', embedding_size=50):

    fname = os.path.join(working_dir, "scdata_embedding")
    if os.path.isfile(fname):
        embedding_model = Word2Vec.load(fname)
        return embedding_model

    train_file = os.path.join(working_dir, "train_data")
    valid_file = os.path.join(working_dir, "valid_data")
    
    train = gen_formatted_data_scdata(train_file)
    valid = gen_formatted_data_scdata(valid_file)
    alldata = train + valid

    embedding_model = Word2Vec(train, size=embedding_size, window=5, min_count=5)
    embedding_model.save(fname)

    return embedding_model
       
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some important parameters.')
    parser.add_argument('-depth', '--embed_size', type=int, default=50, help='embedding_size, default 50.')
    parser.add_argument('-dn', '--dataname', type=str, default="imdbv1", help='generating embeddings for which data: imdbv1 or scdata.')
    
    args = parser.parse_args()
    embed_size = args.embed_size
    dataname = args.dataname.lower()
    assert dataname in ['imdbv1', 'scdata'], "Unknown dataset."

    if dataname == 'imbdv1':
        embedding_model = gen_embedding_model_imdbv1(embedding_size=embed_size)
    else:
        embedding_model = gen_embedding_model_scdata(embedding_size=embed_size)
 
    word1 = "great"
    word2 = "horrible"
    print("similar words of {}:".format(word1))
    print(embedding_model.most_similar('great'))
    print("similar words of {}:".format(word2))
    print(embedding_model.most_similar('horrible'))
