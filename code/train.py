import os
import pandas as pd
import pickle as pkl

from utils import *
from model import *

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

batch_size = 64
n_class = 2
n_epoch = 40
output_dir = "../zzzz"

#=========================== Pre-Load Dataset & Embedding =============================
working_dir = "../data/aclImdb"
train_filename = os.path.join(working_dir, "train_df_file")
test_filename = os.path.join(working_dir, "test_df_file")
emb_filename = os.path.join(working_dir, "emb_matrix")

print("load dataframe for training...")
df_train = pd.read_pickle(train_filename)
print("load dataframe for testing...")
df_test = pd.read_pickle(test_filename)
print("load embedding matrix...")
embed_matrix, word2index_dict, index2word_dict = pkl.load(open(emb_filename, "rb"))
print("generate batches for training...")
data = BucketedDataIterator(df_train, num_buckets=3)

#============================== Build Model ================================
print("build graph...")
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

#============================== Session Config ===============================
saver = tf.train.Saver()
config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

	#========================== Train|Eval ============================
	sess.run(tf.global_variables_initializer())
	print_variables()
	n_batch = int(len(df_train)/batch_size)
	for epoch in range(1, n_epoch+1):
		print("==> epoch %d"%epoch)
		avg_train_loss = 0.0
		avg_train_acc = 0.0
		for i in range(1, n_batch+1):
			x, y, l = data.next_batch(batch_size)
			loss, acc = model.train(sess=sess, batch_X=x, batch_Y=y, batch_L=l)
			avg_train_loss += loss / n_batch
			avg_train_acc += acc / n_batch
			if i%100 == 0:
				print(" %dth batch (%d in total): acc %.4f loss %.4f"%(i, n_batch, acc, loss))
		saver.save(sess, os.path.join(output_dir, "model.ckpt"), epoch)
		print(" avg loss: %.4f" %avg_train_loss)
		print(" avg acc: %.4f" %avg_train_acc)

		if epoch%1 == 0:
			evaluate(model=model, sess=sess, df_eval=df_test)