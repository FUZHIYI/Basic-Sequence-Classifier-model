#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# Get an fixed-learning-rate optimizer instance of, for example, tf.train.AdamOptimizer.
def get_optimizer(opt_type, lr):
	opt_type = opt_type.lower()
	if opt_type in ['sgd', 'gd', 'gradientdescent']:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
	elif opt_type=='adagrad':
		optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
	elif opt_type=='adadelta':
		optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)
	elif opt_type=='adam':
		optimizer = tf.train.AdamOptimizer(learning_rate=lr)
	else:
		assert False, '<Unknown Optimizer> %s'%opt_type
	return optimizer

# Get a certain-type RNNCell instance of, for example, tf.contrib.rnn.BasicLSTMCell.
def get_rnn_cell(cell_type, hidden_size):
	cell_type = cell_type.lower()
	if cell_type in ["rnn", "basicrnn"]:	
		cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
	elif cell_type in ["lstm", "basiclstm"]:
		cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
	elif cell_type == "gru":
		cell = tf.contrib.rnn.GRUCell(hidden_size)
	else:
		assert False, "<Unknown RNN Cell Type>: %s"%cell_type
	return cell

# Input tensor shaped [batch_size, max_time, input_width], return (atten_outs, alphas)
#   atten_ous: an attention tensor shaped [batch_size, input_width] 
#   alphas: an attention weights tensor shaped [batch_size, max_time]
def intra_attention(atten_inputs, atten_size):
	## attention mechanism uses Ilya Ivanov's implementation(https://github.com/ilivans/tf-rnn-attention)
	max_time = int(atten_inputs.shape[1])
	input_width = int(atten_inputs.shape[2])
	W_omega = tf.Variable(tf.random_normal([input_width, atten_size], stddev=0.1, dtype=tf.float32), name="W_omega")
	b_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32), name="b_omega")
	u_omega = tf.Variable(tf.random_normal([atten_size], stddev=0.1, dtype=tf.float32), name="u_omega")
	v = tf.tanh(\
			tf.matmul(tf.reshape(atten_inputs, [-1, input_width]), W_omega) + \
			tf.reshape(b_omega, [1, -1]))
	# u_omega is the summarizing question vector
	vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
	exps = tf.reshape(tf.exp(vu), [-1, max_time])
	alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
	atten_outs = tf.reduce_sum(atten_inputs * tf.reshape(alphas, [-1, max_time, 1]), 1)
	return atten_outs, alphas

# Print Trainable Variables
# MUST used after sess.run(tf.global_variables_initializer()) or sever.file(sess, ckpt)
def print_variables():
	print("[*] Model Trainable Variables:")
	parm_cnt = 0
	variable = [v for v in tf.trainable_variables()]
	#variable = [v for v in tf.global_variables()]
	for v in variable:
		print("   ", v.name, v.get_shape())
		parm_cnt_v = 1
		for i in v.get_shape().as_list():
			parm_cnt_v *= i
		parm_cnt += parm_cnt_v
	print("[*] Model Param Size: %.4fM" %(parm_cnt/1024/1024))

class BucketedDataIterator():
    ## bucketed data iterator uses R2RT's implementation(https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html)
    
    def __init__(self, df, num_buckets=3):
        df = df.sort_values('text_length').reset_index(drop=True)
        # NOTE: sort, http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html
        self.size = int(len(df) / num_buckets)
        self.dfs = []
        for bucket in range(num_buckets):
            self.dfs.append(df.iloc[bucket*self.size: (bucket+1)*self.size])
            # NOTE: slice, http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iloc.html
            # l = list(range(20)); l[19:22]->[19]
        self.num_buckets = num_buckets

        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.shuffle()

        self.epochs = 0

    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            # Note: Return a random sample of items from an axis of object, frac means the ratio: |sample| / items
            self.cursor[i] = 0

    def next_batch(self, n):
        if np.any(self.cursor+n > self.size):
            self.epochs += 1
            self.shuffle()
        i = np.random.randint(0, self.num_buckets)
        res = self.dfs[i].iloc[self.cursor[i]:self.cursor[i]+n]
        self.cursor[i] += n
        if 'sents_length' in res:
        	return np.asarray(res['text'].tolist()), res['label'].tolist(), res['sents_length'].tolist()
        else:
        	return np.asarray(res['text'].tolist()), res['label'].tolist(), res['text_length'].tolist()

# evaluate a whole dataset.
def evaluate(model, sess, df_eval, batch_size=100):

	print(" evaluating dataset sized of %d..." %len(df_eval))
	n_eval_batch = int(len(df_eval)/batch_size)

	total_eval_loss = 0.0
	total_eval_acc = 0.0
	for i in range(0, n_eval_batch):
		batch_x = df_eval['text'][i*batch_size: (i+1)*batch_size].tolist()
		batch_y = df_eval['label'][i*batch_size: (i+1)*batch_size].tolist()
		if 'sents_length' in df_eval:	
			batch_l = df_eval['sents_length'][i*batch_size: (i+1)*batch_size].tolist()
		else:
			batch_l = df_eval['text_length'][i*batch_size: (i+1)*batch_size].tolist()
		eval_loss, eval_acc = model.eval(sess=sess, batch_X=batch_x, batch_L=batch_l, batch_Y=batch_y)
		total_eval_loss += eval_loss * batch_size
		total_eval_acc += eval_acc * batch_size

	n_done = n_eval_batch * batch_size
	n_left = len(df_eval) - n_done
	if n_left > 0:
		batch_x = df_eval['text'][n_done:].tolist()
		batch_y = df_eval['label'][n_done:].tolist()
		batch_l = df_eval['text_length'][n_done:].tolist()
		eval_loss, eval_acc = model.eval(sess=sess, batch_X=batch_x, batch_L=batch_l, batch_Y=batch_y)
		total_eval_loss += eval_loss * n_left
		total_eval_acc += eval_acc * n_left

	print(" avg eval loss: %.4f" %(total_eval_loss/len(df_eval)))
	print(" avg eval acc: %.4f" %(total_eval_acc/len(df_eval)))

# inference a whole dataset.
def inference(model, sess, df_infer, batch_size=100, outfile=None):
	
	if outfile is None:
		outfile = 'inference_output.txt'

	print(" inferencing dataset sized of %d..." %len(df_infer))
	Y_preds = []	
	n_infer_batch = int(len(df_infer)/batch_size)
	for i in range(0, n_infer_batch):
		batch_x = df_infer['text'][i*batch_size: (i+1)*batch_size].tolist()
		if 'sents_length' in df_infer:	
			batch_l = df_infer['sents_length'][i*batch_size: (i+1)*batch_size].tolist()
		else:
			batch_l = df_infer['text_length'][i*batch_size: (i+1)*batch_size].tolist()
		Y_pred = model.infer(sess=sess, batch_X=batch_x, batch_L=batch_l)
		Y_preds += Y_pred[0].tolist()
	n_done = n_infer_batch * batch_size
	n_left = len(df_infer) - n_done
	if n_left > 0:
		batch_x = df_infer['text'][n_done:].tolist()
		batch_l = df_infer['text_length'][n_done:].tolist()
		Y_pred = model.infer(sess=sess, batch_X=batch_x, batch_L=batch_l)
		Y_preds += Y_pred[0].tolist()

	with open(outfile, 'w') as f:
		Y_preds = [str(Yp)+'\n' for Yp in Y_preds]
		f.writelines(Y_preds)
	print(" prediction results have been store in %s"%outfile)
