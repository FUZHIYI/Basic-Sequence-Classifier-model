import os, argparse
import pandas as pd
import pickle as pkl

from utils import *
from model import *

if __name__=="__main__":

	parser = argparse.ArgumentParser(description='Process some important parameters.')
	parser.add_argument('--is_inference', type=bool, default=False, 
		help='if you want to use a trained model to inference, set this flag True.')

	parser.add_argument('--visible_GPU', type=str, default='4', 
		help='choose visible GPUs for CUDA like \'0,3,5\'')
	parser.add_argument('--data_name', type=str, default='imdbv1', 
		help='imdbv1 or scdata.')
	parser.add_argument('--output_dir', type=str, default='../models', 
		help='output dir for saving and restoring your model.')

	parser.add_argument('--model_name', type=str, default='han', 
		help='you could choose: rnn, birnn, han.')
	parser.add_argument('--cell_type', type=str, default='lstm', 
		help='choose RNN Cell type for your model.')
	parser.add_argument('--optimizer_type', type=str, default='adam',
		help='choose your optimizer: adam, sgd, adadelta, adagram')
	parser.add_argument('--learning_rate', type=float, default=1e-4, 
		help='your model\'s learning rate.')
	parser.add_argument('--n_class', type=int, default=2, 
		help='number of classes.')
	parser.add_argument('--hidden_size', type=int, default=128, 
		help='RNN Cell\'s hidden size.')
	parser.add_argument('--batch_size', type=int, default=64, 
		help='batch size for training.')
	parser.add_argument('--n_epoch', type=int, default=40, 
		help='how many epochs you want to train your model.')
	parser.add_argument('--embed_trainable', type=bool, default=False,
		help='whether to fix your embedding layer.')

	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_GPU
	data_name = args.data_name.lower()
	output_dir = args.output_dir.lower()
	is_inference = args.is_inference

	model_name = args.model_name.lower()
	cell_type = args.cell_type.lower()
	optimizer_type = args.optimizer_type.lower()
	learning_rate = args.learning_rate
	n_class = args.n_class
	hidden_size = args.hidden_size
	embed_trainable = args.embed_trainable

	batch_size = args.batch_size
	n_epoch = args.n_epoch

	#=========================== Pre-Load Dataset & Embedding =============================
	if data_name == 'imdbv1':
		working_dir = "../data/aclImdb"
		df_eval = pd.read_pickle(os.path.join(working_dir, "test_df_file"))
		df_infer = None
	elif data_name == 'scdata':
		working_dir = "../data/sentence_consistency_data"
		df_eval = pd.read_pickle(os.path.join(working_dir, "valid_df_file"))
		df_infer = pd.read_pickle(os.path.join(working_dir, "test_df_file"))

	df_train = pd.read_pickle(os.path.join(working_dir, "train_df_file"))
	with open(os.path.join(working_dir, "emb_matrix"), "rb") as f:
		embed_matrix, word2index_dict, index2word_dict = pkl.load(f)
	data = BucketedDataIterator(df_train, num_buckets=3)

	#============================== Build Model ================================
	print("build graph...")
	if model_name == 'han':
		max_textlen, max_seqlen = df_train['text'][0].shape
		model = HierarchicalAttentionClassifier(max_textlen=max_textlen, max_seqlen=max_seqlen, n_class=n_class, embed_matrix=embed_matrix, embed_trainable=embed_trainable, 
			opt_type=optimizer_type, lr=learning_rate, cell_type=cell_type, hidden_size=hidden_size, atten_size=hidden_size)
		#model = HierarchicalAttentionClassifier(max_textlen=max_textlen, max_seqlen=max_seqlen, n_class=n_class, vocab_size=10000, embed_size=50, opt_type=optimizer_type, lr=learning_rate, cell_type=cell_type, hidden_size=hidden_size, atten_size=hidden_size)
	elif model_name == 'rnn':
		(max_seqlen, ) = df_train['text'][0].shape
		model = UniRNNSequenceClassifier(max_seqlen=max_seqlen, n_class=n_class, average_hidden=True, embed_matrix=embed_matrix, embed_trainable=embed_matrix, 
			opt_type=optimizer_type, lr=learning_rate, cell_type=cell_type, hidden_size=hidden_size)
		#model = UniRNNSequenceClassifier(max_seqlen=max_seqlen, n_class=n_class, average_hidden=True, vocab_size=10000, embed_size=50, opt_type=optimizer_type, lr=learning_rate, cell_type=cell_type, hidden_size=hidden_size)
	elif model_name == 'birnn':
		(max_seqlen, ) = df_train['text'][0].shape
		model = BiRNNSequenceClassifier(max_seqlen=max_seqlen, n_class=n_class, average_hidden=True, concat_fw_bw=True, embed_matrix=embed_matrix, embed_trainable=embed_matrix, 
			opt_type=optimizer_type, lr=learning_rate, cell_type=cell_type, hidden_size=hidden_size)
		#model = BiRNNSequenceClassifier(max_seqlen=max_seqlen, n_class=n_class, average_hidden=False, concat_fw_bw=False, vocab_size=10000, embed_size=50, opt_type=optimizer_type, lr=learning_rate, cell_type=cell_type, hidden_size=hidden_size)
	else:
		assert False, "Unknown or not supportive models: %s"%model_name

	#=============================== Config ==================================
	saver = tf.train.Saver()
	config = tf.ConfigProto()
	config.allow_soft_placement = True
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		
		#=========================== Infer ==================================
		if is_inference:
			latest_ckpt_file = tf.train.latest_checkpoint(output_dir)
			print(" Load model from the lateset checkpoint: %s"%latest_ckpt_file)
			saver.restore(sess, latest_ckpt_file)
			resume_from_epoch = int(str(latest_ckpt_file).split('-')[1])
			if df_infer is not None:
				inference(model=model, sess=sess, df_infer=df_infer)
			exit(0)

		#=========================== Train ==================================
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

			# evaluating after trainning every epoch data.
			if epoch%1 == 0:
				evaluate(model=model, sess=sess, df_eval=df_eval)

		# aftering training, inference.
		if df_infer is not None:
			inference(model=model, sess=sess, df_infer=df_infer)