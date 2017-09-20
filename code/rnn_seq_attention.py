from random import shuffle
import tensorflow as tf
import nn_mod as nn
import numpy as np
import pickle
import sys

'''
Recurrent neural network architecture using sequential layers of LSTM units, and soft attention 
based on https://arxiv.org/pdf/1509.06664.pdf
'''


# paths
glove_dims = "100"
embed_path = 'obj/emb_train_'+glove_dims+'.pkl'

data_path = '/Users/punit/stat-nlp-book/data/nn/'

data_path = 'data/'

# load vocabulary dict and embeddings matrix
with open(embed_path, 'rb') as handle:
    embeds = pickle.load(handle)

# load data
data_train = nn.load_corpus(data_path + "train.tsv")
data_dev = nn.load_corpus(data_path + "dev.tsv")
assert(len(data_train) == 45502)

# find maximum sentence length in training set
train_stories, _, vocab, _ = nn.mod_pipeline(data_train)
max_sent_len = train_stories.shape[2]


### MODEL CONFIGURATION ###
trainable = True               # trainable embeddings
BATCH_SIZE = 25                 # batch size
EPOCHS = 5                     # epochs
num_units = 40                    # number of units in each LSTMCell
num_layers = 1                     # number of stacked LSTMs
KEEP_PRB_1 = 0.99                    # dropout probability of keeping value
_units = 60                         # output RNN parameters
_layers = 1                         # output RNN parameters
KEEP_PRB_2 = 0.9                    # dropout probability of keeping value
learning_rate = 0.001            # optimizer learning rate
target_size = 5                 # orderings for 5 sentences
vocab_size = len(vocab)         # size of vocabulary


seq_story = tf.placeholder(tf.int64, [None, None, None], "story")        # [batch_size x 5 x max_seq_length]
seq_order = tf.placeholder(tf.int64, [None, None], "order")              # [batch_size x 5]
seq_lens = tf.placeholder(tf.int64, [None, None], "seq_lens")     # [batch_size x 5]
batch_size = tf.shape(seq_story)[0]
keep_prob = tf.placeholder(tf.float64)          
keep_prob_2 = tf.placeholder(tf.float64)        

with tf.variable_scope("seq"):
    # Word embeddings
    sentences = [tf.reshape(x, [batch_size, -1]) for x in tf.split(1, 5, seq_story)]  # 5 times [batch_size x max_sent_len]
    embeddings = tf.get_variable("embeddings", initializer=embeds, trainable=trainable)
    inputs = [tf.nn.embedding_lookup(embeddings, sentence)   # 5 times [batch_size x max_sent_len x embedding_size]
                          for sentence in sentences]

with tf.variable_scope("lstms") as varscope:
    # first LSTM
    index = 0
    lstm1 = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True, activation=tf.nn.relu6)
    lstm1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, output_keep_prob=keep_prob)
    lstm1 = tf.nn.rnn_cell.MultiRNNCell([lstm1] * num_layers)
    out1, state1 = tf.nn.dynamic_rnn(lstm1, inputs[index], dtype=tf.float64, initial_state=None, sequence_length=seq_lens[:,index])
    varscope.reuse_variables()

    # second LSTM
    index = 1
    lstm2 = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True, activation=tf.nn.relu6)
    lstm2 = tf.nn.rnn_cell.DropoutWrapper(lstm2, output_keep_prob=keep_prob)
    lstm2 = tf.nn.rnn_cell.MultiRNNCell([lstm2] * num_layers)
    out2, state2 = tf.nn.dynamic_rnn(lstm2, inputs[index], dtype=tf.float64, initial_state=state1, sequence_length=seq_lens[:,index])
    varscope.reuse_variables()

    # third LSTM
    index = 2
    lstm3 = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True, activation=tf.nn.relu6)
    lstm3 = tf.nn.rnn_cell.DropoutWrapper(lstm3, output_keep_prob=keep_prob)
    lstm3 = tf.nn.rnn_cell.MultiRNNCell([lstm3] * num_layers)
    out3, state3 = tf.nn.dynamic_rnn(lstm3, inputs[index], dtype=tf.float64, initial_state=state2, sequence_length=seq_lens[:,index])
    varscope.reuse_variables()

    # fourth LSTM
    index = 3
    lstm4 = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True, activation=tf.nn.relu6)
    lstm4 = tf.nn.rnn_cell.DropoutWrapper(lstm4, output_keep_prob=keep_prob)
    lstm4 = tf.nn.rnn_cell.MultiRNNCell([lstm4] * num_layers)
    out4, state4 = tf.nn.dynamic_rnn(lstm4, inputs[index], dtype=tf.float64, initial_state=state3, sequence_length=seq_lens[:,index])
    varscope.reuse_variables()

    # last LSTM
    index = 4
    lstm5 = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True, activation=tf.nn.relu6)
    lstm5 = tf.nn.rnn_cell.DropoutWrapper(lstm5, output_keep_prob=keep_prob)
    lstm5 = tf.nn.rnn_cell.MultiRNNCell([lstm5] * num_layers)
    out5, state5 = tf.nn.dynamic_rnn(lstm5, inputs[index], dtype=tf.float64, initial_state=state4, sequence_length=seq_lens[:,index])
    '''
    out dimensions: [batch_size x max_sent_len x num_units]
    state dimensions: num_layers times [batch_size x num_units]
    '''

### attention based on https://arxiv.org/pdf/1509.06664.pdf ###
def attention(out1, s2, B, L, k, curr_scope):
    with tf.variable_scope(curr_scope):
        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        W_y = tf.get_variable("W_y", shape=[k, k], initializer=initializer, trainable=True, dtype=tf.float64) # [k x k]
        W_h = tf.get_variable("W_h", shape=[k, k], initializer=initializer, trainable=True, dtype=tf.float64) # [k x k]
        W_p = tf.get_variable("W_p", shape=[k, k], initializer=initializer, trainable=True, dtype=tf.float64) # [k x k]
        W_x = tf.get_variable("W_x", shape=[k, k], initializer=initializer, trainable=True, dtype=tf.float64) # [k x k]

        w = tf.get_variable("w", shape=[1,k], initializer=initializer, trainable=True, dtype=tf.float64) # [1 x k]

        out1_t = tf.transpose(out1, perm=[2,0,1])  # [k x batch_size x L]
        Y = tf.reshape(out1_t, [k, -1])   # [k  x (L*batch_size)]
        left = tf.matmul(W_y, Y) # [k x (batch_size*L)]
        left = tf.reshape(left, [k, B, L])
        left = tf.transpose(left, perm=[1, 0, 2])  # [batch_size x k x L]

        hN = s2.h # [batch_size x k]
        right = tf.matmul(hN, W_h) # [batch_size x k]
        right = tf.expand_dims(right, axis=2) # [batch_size x k x 1]

        M = tf.tanh(left + right)  # [batch_size x k x L]

        M = tf.transpose(M, perm=[1,0,2])
        M = tf.reshape(M, [k, -1])  #[k x (L*batch_size)]
        wM = tf.matmul(w, M)  # [1 x (L*batch_size)]
        wM = tf.reshape(wM, [1, B, L]) # [1 x batch_size x L]
        wM = tf.transpose(wM, perm=[1,0,2])  # [batch_size  x 1 x L]

        alpha = tf.nn.softmax(wM, dim=-1)  # [batch_size x 1 x L]

        r = tf.batch_matmul(alpha, out1)  # [batch_size x 1 x k]

        r = tf.transpose(r, perm=[2,0,1])  # [k x batch_size x 1]
        r = tf.reshape(r, [k,-1])  # [k x batch_size]
        Wpr = tf.matmul(W_p, r)  # [k x batch_size]
        Wpr = tf.transpose(Wpr) # [batch_size x k]

        Wxhn = tf.matmul(hN, W_x) # [batch_size x k]

        h_final = tf.tanh(Wpr + Wxhn)  # [batch_size x k]

        return h_final


# attention output vectors each of dimension [batch_size x num_units]
h_0 = state1[-1].h
h_1 = attention(out1, state2[-1], batch_size, tf.shape(out1)[1], num_units, "att1")
h_2 = attention(out2, state3[-1], batch_size, tf.shape(out2)[1], num_units, "att2")
h_3 = attention(out3, state4[-1], batch_size, tf.shape(out3)[1], num_units, "att3")
h_4 = attention(out4, state5[-1], batch_size, tf.shape(out4)[1], num_units, "att4")

with tf.variable_scope("seq"):
    # create final input tensor for bidirectional RNN
    new_inputs = [h_0, h_1, h_2, h_3, h_4]  # 5 times [batch_size x num_units]
    sl = len(new_inputs)
    new_inputs = [tf.expand_dims(x,axis=1) for x in new_inputs]
    new_inputs = tf.concat(1, new_inputs)  # [batch_size x 5 x num_units]

    # final LSTM
    lstm_cell = tf.nn.rnn_cell.LSTMCell(_units, state_is_tuple=True, activation=tf.nn.relu6)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob_2)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * _layers, state_is_tuple=True)
    # final RNN
    final_outputs, final_state = tf.nn.dynamic_rnn(cell, new_inputs, dtype=tf.float64) 
    output = tf.reshape(final_outputs, [-1, 5*_units])  # [batch_size x (5*_units)]


with tf.variable_scope("seq"):
    # hidden layer
    H1 = tf.nn.relu6(tf.contrib.layers.linear(output, 150))

    # final linear transformation
    logits_flat = tf.contrib.layers.linear(H1, 5*target_size)  # [batch_size x 5*target_size]

    # unflatten logits (need this shape for sparse softmax)
    logits = tf.reshape(logits_flat, [-1, 5, target_size]) # dimensions: [batch_size x 5 x target_size]

    # cross entropy loss function
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, seq_order))
    tf.summary.scalar('cross_entropy', loss)


# optimizer
seq_train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# prediction function
seq_unpacked_logits = [tensor for tensor in tf.unpack(logits, axis=1)]
seq_softmaxes = [tf.nn.softmax(tensor) for tensor in seq_unpacked_logits]
seq_softmaxed_logits = tf.pack(seq_softmaxes, axis=1)
seq_predict = tf.arg_max(seq_softmaxed_logits, 2)

# accuracy 
seq_correct = tf.equal(seq_predict, seq_order)
seq_accuracy = tf.reduce_mean(tf.cast(seq_correct, tf.float32))
tf.summary.scalar('accuracy', seq_accuracy)


# create Session
sess = tf.Session()
# initialize variables
sess.run(tf.initialize_all_variables())

# merge summaries
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('logs/train', sess.graph)
test_writer = tf.train.SummaryWriter('logs/test')


# get development set and create dev dictionary
dev_stories, dev_orders, _, dev_seq_lens = nn.mod_pipeline(data_dev, vocab=vocab, 
    max_sent_len_=max_sent_len)
dev_feed_dict = {seq_story: dev_stories, seq_order: dev_orders, seq_lens: dev_seq_lens, keep_prob: 1.0, keep_prob_2: 1.0}


for epoch in range(EPOCHS):
    # Setup for training epoch
    shuffle(data_train)   # randomly shuffle training set --> natural random batches
    train_stories, train_orders, _, train_seq_lens = nn.mod_pipeline(data_train, vocab=vocab, 
        shuffle_sentences=(epoch > 0))

    # chunks: number of batches to cover entire data set
    n = train_stories.shape[0]
    chunks = n // BATCH_SIZE

    if epoch == 2:
        KEEP_PRB_1 -= 0.15
        KEEP_PRB_2 -= 0.2

    print('----- Epoch', epoch, '-----')
    for i in range(chunks):
        if i % 100 == 0:
            # test summary
            summary, dev_accuracy = sess.run([merged, seq_accuracy], feed_dict=dev_feed_dict)
            test_writer.add_summary(summary, (epoch * chunks) + i)
            print(' Dev accuracy:', dev_accuracy)
            sys.stdout.flush()

        # training step + training summary
        inds = slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE, 1)
        summary, _ = sess.run([merged, seq_train_step], 
            feed_dict={seq_story: train_stories[inds], seq_order: train_orders[inds], seq_lens: train_seq_lens[inds], 
            keep_prob: KEEP_PRB_1, keep_prob_2: KEEP_PRB_2})
        train_writer.add_summary(summary, (epoch * chunks) + i)

    print(' Dev accuracy:', sess.run(seq_accuracy, feed_dict=dev_feed_dict))

nn.save_model(sess)




