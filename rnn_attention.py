from random import shuffle
import tensorflow as tf
import nn_mod as nn
import numpy as np
import pickle

'''
Recurrent neural network architecture using layers of LSTM units, and soft attention 
based on https://arxiv.org/pdf/1509.06664.pdf
'''


# paths
vocab_path = 'obj/vocab.pickle'
embed_path = 'obj/embeddings.pickle'
data_path = '/Users/punit/stat-nlp-book/data/nn/'

#data_path = 'data/'


# load vocabulary dict and embeddings matrix
with open(vocab_path, 'rb') as handle:
    vocab = pickle.load(handle)
with open(embed_path, 'rb') as handle:
    embeds = pickle.load(handle)

# load data
data_train = nn.load_corpus(data_path + "train.tsv")
data_dev = nn.load_corpus(data_path + "dev.tsv")
assert(len(data_train) == 45502)

# convenience switches (if false, model is loaded)
TRAIN_SEQ = True
TRAIN_MASTER = True

#---------------------------------------------------------

# GLOBAL CONFIGURATION

BATCH_SIZE = 25                # batch size
learning_rate = 1e-3            # optimizer learning rate
target_size = 5                 # orderings for 5 sentences
vocab_size = len(vocab)         # size of vocabulary

#---------------------------------------------------------

# SEQUENTIAL MODEL

# find maximum sentence length in training set
seq_train_stories, _, _, _ = nn.mod_pipeline(data_train, vocab=vocab)
max_sent_len = seq_train_stories.shape[2]


# SEQUENTIAL CONFIGURATION 

SEQ_EPOCHS = 1                     # EPOCHS
seq_num_units = 20                    # number of units in each LSTMCell
seq_num_layers = 2                     # number of stacked LSTMs
SEQ_KEEP_PRB = 0.9                    # dropout probability of keeping value
bidirectional = True           # enable bidirectional output layer


seq_story = tf.placeholder(tf.int64, [None, None, None], "seq_story")        # [seq_batch_size x 5 x max_seq_length]
seq_order = tf.placeholder(tf.int64, [None, None], "seq_order")              # [seq_batch_size x 5]
seq_lens = tf.placeholder(tf.int64, [None, None], "seq_lens")     # [seq_batch_size x 5]
seq_batch_size = tf.shape(seq_story)[0]
seq_keep_prob = tf.placeholder(tf.float64)          # dropout probability placeholder

with tf.variable_scope("seq"):
    # Word embeddings
    sentences = [tf.reshape(x, [seq_batch_size, -1]) for x in tf.split(1, 5, seq_story)]  # 5 times [seq_batch_size x max_sent_len]
    embeddings = tf.get_variable("embeddings", initializer=embeds, trainable=True)
    inputs = [tf.nn.embedding_lookup(embeddings, sentence)   # 5 times [seq_batch_size x max_sent_len x embedding_size]
                          for sentence in sentences]

with tf.variable_scope("lstms") as varscope:
    # first LSTM
    index = 0
    lstm1 = tf.nn.rnn_cell.LSTMCell(seq_num_units, state_is_tuple=True)
    lstm1 = tf.nn.rnn_cell.MultiRNNCell([lstm1] * seq_num_layers)
    lstm1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, output_keep_prob=seq_keep_prob)
    out1, state1 = tf.nn.dynamic_rnn(lstm1, inputs[index], dtype=tf.float64, initial_state=None, sequence_length=seq_lens[:,index])
    varscope.reuse_variables()

    # second LSTM
    index = 1
    lstm2 = tf.nn.rnn_cell.LSTMCell(seq_num_units, state_is_tuple=True)
    lstm2 = tf.nn.rnn_cell.MultiRNNCell([lstm2] * seq_num_layers)
    lstm2 = tf.nn.rnn_cell.DropoutWrapper(lstm2, output_keep_prob=seq_keep_prob)
    out2, state2 = tf.nn.dynamic_rnn(lstm2, inputs[index], dtype=tf.float64, initial_state=state1, sequence_length=seq_lens[:,index])
    varscope.reuse_variables()

    # third LSTM
    index = 2
    lstm3 = tf.nn.rnn_cell.LSTMCell(seq_num_units, state_is_tuple=True)
    lstm3 = tf.nn.rnn_cell.MultiRNNCell([lstm3] * seq_num_layers)
    lstm3 = tf.nn.rnn_cell.DropoutWrapper(lstm3, output_keep_prob=seq_keep_prob)
    out3, state3 = tf.nn.dynamic_rnn(lstm3, inputs[index], dtype=tf.float64, initial_state=state2, sequence_length=seq_lens[:,index])
    varscope.reuse_variables()

    # fourth LSTM
    index = 3
    lstm4 = tf.nn.rnn_cell.LSTMCell(seq_num_units, state_is_tuple=True)
    lstm4 = tf.nn.rnn_cell.MultiRNNCell([lstm4] * seq_num_layers)
    lstm4 = tf.nn.rnn_cell.DropoutWrapper(lstm4, output_keep_prob=seq_keep_prob)
    out4, state4 = tf.nn.dynamic_rnn(lstm4, inputs[index], dtype=tf.float64, initial_state=state3, sequence_length=seq_lens[:,index])
    varscope.reuse_variables()

    # last LSTM
    index = 4
    lstm5 = tf.nn.rnn_cell.LSTMCell(seq_num_units, state_is_tuple=True)
    lstm5 = tf.nn.rnn_cell.MultiRNNCell([lstm5] * seq_num_layers)
    lstm5 = tf.nn.rnn_cell.DropoutWrapper(lstm5, output_keep_prob=seq_keep_prob)
    out5, state5 = tf.nn.dynamic_rnn(lstm5, inputs[index], dtype=tf.float64, initial_state=state4, sequence_length=seq_lens[:,index])
    '''
    out dimensions: [seq_batch_size x max_sent_len x seq_num_units]
    state dimensions: seq_num_layers times [seq_batch_size x seq_num_units]
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

        out1_t = tf.transpose(out1, perm=[2,0,1])  # [k x seq_batch_size x L]
        Y = tf.reshape(out1_t, [k, -1])   # [k  x (L*seq_batch_size)]
        left = tf.matmul(W_y, Y) # [k x (seq_batch_size*L)]
        left = tf.reshape(left, [k, B, L])
        left = tf.transpose(left, perm=[1, 0, 2])  # [seq_batch_size x k x L]

        hN = s2.h # [seq_batch_size x k]
        right = tf.matmul(hN, W_h) # [seq_batch_size x k]
        right = tf.expand_dims(right, axis=2) # [seq_batch_size x k x 1]

        M = tf.tanh(left + right)  # [seq_batch_size x k x L]

        M = tf.transpose(M, perm=[1,0,2])
        M = tf.reshape(M, [k, -1])  #[k x (L*seq_batch_size)]
        wM = tf.matmul(w, M)  # [1 x (L*seq_batch_size)]
        wM = tf.reshape(wM, [1, B, L]) # [1 x seq_batch_size x L]
        wM = tf.transpose(wM, perm=[1,0,2])  # [seq_batch_size  x 1 x L]

        alpha = tf.nn.softmax(wM, dim=-1)  # [seq_batch_size x 1 x L]

        r = tf.batch_matmul(alpha, out1)  # [seq_batch_size x 1 x k]

        r = tf.transpose(r, perm=[2,0,1])  # [k x seq_batch_size x 1]
        r = tf.reshape(r, [k,-1])  # [k x seq_batch_size]
        Wpr = tf.matmul(W_p, r)  # [k x seq_batch_size]
        Wpr = tf.transpose(Wpr) # [seq_batch_size x k]

        Wxhn = tf.matmul(hN, W_x) # [seq_batch_size x k]

        h_final = tf.tanh(Wpr + Wxhn)  # [seq_batch_size x k]

        return h_final

# attention output vectors each of dimension [seq_batch_size x seq_num_units]
h_1 = attention(out1, state2[-1], seq_batch_size, tf.shape(out1)[1], seq_num_units, "att1")
h_2 = attention(out2, state3[-1], seq_batch_size, tf.shape(out2)[1], seq_num_units, "att2")
h_3 = attention(out3, state4[-1], seq_batch_size, tf.shape(out3)[1], seq_num_units, "att3")
h_4 = attention(out4, state5[-1], seq_batch_size, tf.shape(out4)[1], seq_num_units, "att4")

with tf.variable_scope("seq"):
    # create final input tensor for bidirectional RNN
    new_inputs = [h_1, h_2, h_3, h_4]  # 4 times [seq_batch_size x seq_num_units]
    sl = len(new_inputs)
    new_inputs = [tf.expand_dims(x,axis=1) for x in new_inputs]
    new_inputs = tf.concat(1, new_inputs)  # [seq_batch_size x 4 x seq_num_units]

    # output RNN parameters
    _units = 20
    _layers = 1

    # final LSTM
    lstm_cell = tf.nn.rnn_cell.LSTMCell(_units, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * _layers, state_is_tuple=True)
    # final RNN
    if bidirectional:
        _seq_len = tf.fill(tf.expand_dims(seq_batch_size, 0), tf.constant(sl, dtype=tf.int64)) 
        final_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, 
            inputs=new_inputs, dtype=tf.float64, sequence_length=_seq_len) 
        # hidden vectors at final step (combined forward and reverse LSTMs)
        output = tf.concat(1, [final_state[0][-1].h, final_state[1][-1].h])  # [seq_batch_size x (2*_units)]
    else:
        final_outputs, final_state = tf.nn.dynamic_rnn(cell, new_inputs, dtype=tf.float64) 
        # hidden vectors at final step 
        output = final_state[-1].h  # [seq_batch_size x _units]

with tf.variable_scope("seq"):
    # final linear transformation
    logits_flat = tf.contrib.layers.linear(output, 5*target_size)  # [seq_batch_size x 5*target_size]

    # unflatten logits (need this shape for sparse softmax)
    logits = tf.reshape(logits_flat, [-1, 5, target_size]) # dimensions: [seq_batch_size x 5 x target_size]

    # cross entropy loss function
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, seq_order))

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
    tf.summary.scalar('seq_accuracy', seq_accuracy)

#---------------------------------------------------------

# MASTER MODEL

# find maximum sentence length in training set
train_stories, train_orders, _ = nn.pipeline(data_train, vocab=vocab)
max_seq_len = train_stories.shape[2]


# MASTER MODEL CONFIGURATION

EPOCHS = 1                     # epochs
num_units = 20                    # number of units in each LSTMCell
num_layers = 2                     # number of stacked LSTMs
KEEP_PRB = 0.9                    # dropout probability of keeping value
num_steps = 5 * max_seq_len         # time steps = max sequence length


story = tf.placeholder(tf.int64, [None, None, None], "story")        # [batch_size x 5 x max_seq_length]
order = tf.placeholder(tf.int64, [None, None], "order")              # [batch_size x 5]
batch_size = tf.shape(story)[0]
keep_prob = tf.placeholder(tf.float64) # dropout probability placeholder

with tf.variable_scope("master"):
    # Word embeddings
    sentences = [tf.reshape(x, [batch_size, -1]) for x in tf.split(1, 5, story)]  # 5 times [batch_size x max_length]
    embeddings = tf.get_variable("embeddings", initializer=embeds, trainable=True)
    sentences_embedded = [tf.nn.embedding_lookup(embeddings, sentence)   # [batch_size x max_seq_length x embedding_size]
                          for sentence in sentences]
    # combine 5 sentences into one long sequence 
    inputs = tf.concat(1, sentences_embedded) # [batch_size x (5 * max_seq_length) x embedding_size]


with tf.variable_scope("master"):
    # RNN architecture
    cells = []
    for i in range(num_layers):
        cells.append(tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True))
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float64) 
    '''
    outputs dimensions: [batch_size x total_sequence_length x num_units]
    state dimensions: [batch_size x tuple(cell.state_size)]
    '''

    # reshape to 2D tensor
    output = tf.reshape(outputs, [-1, num_steps * num_units])

    # linear transformation 
    W = tf.get_variable("W", [num_steps*num_units, 5 * target_size], dtype=tf.float64)
    b = tf.get_variable("b", [5 * target_size], dtype=tf.float64)
    logits_flat = tf.matmul(output, W) + b  # dimensions: [batch_size x (5 * target_size)]

    # unflatten logits (need this shape for sparse softmax)
    logits = tf.reshape(logits_flat, [-1, 5, target_size]) # dimensions: [batch_size x 5 x target_size]

    # cross entropy loss function
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, order))
    tf.summary.scalar('cross_entropy', loss)

    # optimizer
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # prediction function
    master_unpacked_logits = [tensor for tensor in tf.unpack(logits, axis=1)]
    master_softmaxes = [tf.nn.softmax(tensor) for tensor in master_unpacked_logits]
    master_softmaxed_logits = tf.pack(master_softmaxes, axis=1)
    master_predict = tf.arg_max(master_softmaxed_logits, 2)

    # accuracy 
    master_correct = tf.equal(master_predict, order)
    master_accuracy = tf.reduce_mean(tf.cast(master_correct, tf.float32))
    tf.summary.scalar('accuracy', master_accuracy)

#---------------------------------------------------------

# MODEL AVERAGING

master = 0.5
sequential = 0.5
softmaxed_logits = (sequential * seq_softmaxed_logits) + (master * master_softmaxed_logits)
predict = tf.arg_max(softmaxed_logits, 2)


#---------------------------------------------------------

# CREATE SESSION

# create Session
sess = tf.Session()
# initialize variables
sess.run(tf.initialize_all_variables())

# merge summaries
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('logs/train', sess.graph)
test_writer = tf.train.SummaryWriter('logs/test')

# create saver
saver = tf.train.Saver()

# dummy placeholders
storss = np.zeros([3,5,2])
ordss = np.zeros([3,5])
seqqss = np.zeros([3,5])

#---------------------------------------------------------

# TRAIN SEQUENTIAL MODEL

if TRAIN_SEQ:
    # get development set and create dev dictionary
    seq_dev_stories, seq_dev_orders, _, dev_seq_lens = nn.mod_pipeline(data_dev, vocab=vocab, max_sent_len_=max_sent_len)
    seq_dev_feed_dict = {seq_story: seq_dev_stories, seq_order: seq_dev_orders, seq_lens: dev_seq_lens, seq_keep_prob: 1.0,
        story: storss, order: ordss, keep_prob: 1.0}

    # chunks: number of batches to cover entire data set
    seq_n = seq_train_stories.shape[0]
    chunks = seq_n // BATCH_SIZE

    for epoch in range(SEQ_EPOCHS):
        # Setup for training epoch
        shuffle(data_train)   # randomly shuffle training set --> natural random batches
        seq_train_stories, seq_train_orders, _, train_seq_lens = nn.mod_pipeline(data_train, vocab=vocab)

        print('----- Epoch', epoch, '-----')
        for i in range(chunks):
            if i % 100 == 0:
                # test summary
                summary, dev_accuracy = sess.run([merged, seq_accuracy], feed_dict=seq_dev_feed_dict)
                test_writer.add_summary(summary, (epoch * chunks) + i)
                print(' Dev accuracy:', dev_accuracy)

            # training step + training summary
            inds = slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE, 1)
            summary, _ = sess.run([merged, seq_train_step], 
                feed_dict={seq_story: seq_train_stories[inds], 
                    seq_order: seq_train_orders[inds], seq_lens: train_seq_lens[inds], seq_keep_prob: SEQ_KEEP_PRB,
                    story: storss, order: ordss, keep_prob: 1.0})
            train_writer.add_summary(summary, (epoch * chunks) + i)

        print(' Dev accuracy:', sess.run(seq_accuracy, feed_dict=seq_dev_feed_dict))

    nn.save_model(sess, "seqmodel")
else:
    saver.restore(sess, './model/model.checkpoint_seqmodel')


#---------------------------------------------------------

# TRAIN MASTER MODEL

if TRAIN_MASTER:
    # chunks: number of batches to cover entire data set
    n = train_stories.shape[0]
    chunks = n // BATCH_SIZE

    # get dev data and create development set dict
    dev_stories, dev_orders, _ = nn.pipeline(data_dev, vocab=vocab, max_sent_len_=max_seq_len)
    dev_feed_dict = {story: dev_stories, order: dev_orders, keep_prob: 1.0,
        seq_story: storss, seq_order: ordss, seq_lens: seqqss, seq_keep_prob: 1.0}

    for epoch in range(EPOCHS):
        # Setup for training epoch
        shuffle(data_train)   # randomly shuffle training set --> natural random batches
        train_stories, train_orders, _ = nn.pipeline(data_train, vocab=vocab)

        print('----- Epoch', epoch, '-----')
        for i in range(chunks):
            if i % 100 == 0:
                # test summary
                summary, dev_accuracy = sess.run([merged, master_accuracy], feed_dict=dev_feed_dict)
                test_writer.add_summary(summary, (epoch * chunks) + i)
                print(' Dev accuracy:', dev_accuracy)

            # training step + training summary
            inds = slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE, 1)
            summary, _ = sess.run([merged, train_step], 
                feed_dict={story: train_stories[inds], order: train_orders[inds], keep_prob: KEEP_PRB,
                    seq_story: storss, seq_order: ordss, seq_lens: seqqss, seq_keep_prob: 1.0})
            train_writer.add_summary(summary, (epoch * chunks) + i)

        print(' Dev accuracy:', sess.run(master_accuracy, feed_dict=dev_feed_dict))

    nn.save_model(sess, "mastermodel")
else:
    saver.restore(sess, './model/model.checkpoint_mastermodel')

#---------------------------------------------------------

# FINAL PREDICTIONS

print('Sequential model final accuracy:', sess.run(seq_accuracy, feed_dict=seq_dev_feed_dict))
print('Master model final accuracy:', sess.run(master_accuracy, feed_dict=dev_feed_dict))

# dev dicitonary
final_feed_dict = {story: dev_stories, order: dev_orders, keep_prob: 1.0, 
    seq_story: seq_dev_stories, seq_order: seq_dev_orders, seq_lens: dev_seq_lens}

# make predictions
dev_predicted = sess.run(predict, feed_dict=final_feed_dict)

print("Averaged prediction accuracy:")
print(nn.calculate_accuracy(dev_orders, dev_predicted))
























