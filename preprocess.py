import numpy as np
import random
import os
import re
import tensorflow as tf

'''

Text preprocessing pipeline.

'''


# order representation
def order_representation(data):
    orders = []
    for instance in data:
        ordering = data['order']


# data loading
def load_corpus(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            splits = [x.strip() for x in line.split("\t")]
            current_story = splits[0:5]
            current_order = list(int(elem) for elem in splits[5:])
            instance = {"story": current_story, "order": current_order}
            data.append(instance)
    return data


token = re.compile("[\w-]+|'m|'t|'ll|'ve|'d|'s|\'")
def tokenize(input):
    return [word.lower() for word in token.findall(input)]


# preprocessing pipeline, used to load the data intro a structure required by the model
def pipeline(data, vocab=None, max_sent_len_=None, shuffle_sentences=False):
    is_ext_vocab = True
    if vocab is None:
        is_ext_vocab = False
        vocab = {'<PAD>': 0, '<OOV>': 1}

    max_sent_len = -1
    data_sentences = []
    data_orders = []
    for instance in data:
        sents = []
        for sentence in instance['story']:
            sent = []
            tokenized = tokenize(sentence)
            for token in tokenized:
                if not is_ext_vocab and token not in vocab:
                    vocab[token] = len(vocab)
                if token not in vocab:
                    token_id = vocab['<OOV>']
                else:
                    token_id = vocab[token]
                sent.append(token_id)
            if len(sent) > max_sent_len:
                max_sent_len = len(sent)
            sents.append(sent)
        # random permutation of story order
        if shuffle_sentences:
            zipped = list(zip(sents, instance['order']))
            random.shuffle(zipped)
            a,b = zip(*zipped)
            data_sentences.append(a)
            data_orders.append(b)
        else:
            data_sentences.append(sents)
            data_orders.append(instance['order'])

    if max_sent_len_ is not None:
        max_sent_len = max_sent_len_
    out_sentences = np.full([len(data_sentences), 5, max_sent_len], vocab['<PAD>'], dtype=np.int32)

    for i, elem in enumerate(data_sentences):
        for j, sent in enumerate(elem):
            out_sentences[i, j, 0:len(sent)] = sent

    out_orders = np.array(data_orders, dtype=np.int32)

    return out_sentences, out_orders, vocab



# displaying the loaded data
def show_data_instance(data_stories, data_orders, vocab, num_story):
    inverted_vocab = {value: key for key, value in vocab.items()}
    print('Input:\n Story:')
    story_example = {}
    for i, elem in enumerate(data_stories[num_story]):
        x = list(inverted_vocab[ch] if ch in inverted_vocab else '<OOV>'
                 for ch in elem if ch != 0)
        story_example[data_orders[num_story][i]] = " ".join(x)
        print(' '," ".join(x))
    print(' Order:\n ', data_orders[num_story])
    print('\nDesired story:')
    for (k, v) in sorted(story_example.items()):
        print(' ',v)


# accuracy calculation
def calculate_accuracy(orders_gold, orders_predicted):
    num_correct = np.sum(orders_predicted == orders_gold)
    num_total =  orders_gold.shape[0] * 5
    return num_correct / num_total


# save the model params to the hard drive
def save_model(session, name):
    if not os.path.exists('./model/'):
        os.mkdir('./model/')
    saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)
    saver.save(session, './model/model.checkpoint_'+name)

def save_model_with_name(session, name):
    if not os.path.exists('./model/'):
        os.mkdir('./model/')
    saver = tf.train.Saver()
    s = './model/model-' + name + '.highscoresave'
    saver.save(session, s)

    # preprocessing pipeline, used to load the data intro a structure required by the model
def seq_mod_pipeline(data, vocab=None, max_seq_len_=None):
    is_ext_vocab = True
    if vocab is None:
        is_ext_vocab = False
        vocab = {'<PAD>' : 0, '<OOV>': 1, '<DELIM>': 2}

    max_seq_len = -1
    data_sentences = []
    data_orders = []
    for instance in data:
        sents = []
        for sentence in instance['story']:
            sent = []
            tokenized = tokenize(sentence)
            for token in tokenized:
                if not is_ext_vocab and token not in vocab:
                    vocab[token] = len(vocab)
                if token not in vocab:
                    token_id = vocab['<OOV>']
                else:
                    token_id = vocab[token]
                sent.append(token_id)
            sent.append(vocab['<DELIM>'])
            sents.extend(sent)

        max_seq_len = max(max_seq_len, len(sents))

        data_sentences.append(sents)
        data_orders.append(instance['order'])

    if max_seq_len_ is not None:
        max_seq_len = max_seq_len_
    out_sentences = np.full([len(data_sentences), max_seq_len], vocab['<PAD>'], dtype=np.int32)

    seq_lens = []
    for i, seq in enumerate(data_sentences):
        out_sentences[i, 0:len(seq)] = seq
        seq_lens.append(len(seq))

    out_orders = np.array(data_orders, dtype=np.int32)

    return out_sentences, out_orders, vocab, seq_lens



def mod_pipeline(data, vocab=None, max_sent_len_=None, shuffle_sentences=False):
    is_ext_vocab = True
    if vocab is None:
        is_ext_vocab = False
        vocab = {'<PAD>': 0, '<OOV>': 1, '<DELIM>': 2}

    max_sent_len = -1
    data_sentences = []
    data_orders = []
    for instance in data:
        sents = []
        for ii, sentence in enumerate(instance['story']):
            sent = []
            if ii > 0:
                sent.append(vocab['<DELIM>'])
            tokenized = tokenize(sentence)
            for token in tokenized:
                if not is_ext_vocab and token not in vocab:
                    vocab[token] = len(vocab)
                if token not in vocab:
                    token_id = vocab['<OOV>']
                else:
                    token_id = vocab[token]
                sent.append(token_id)
            if len(sent) > max_sent_len:
                max_sent_len = len(sent)
            sents.append(sent)
        # random permutation of story order
        if shuffle_sentences:
            zipped = list(zip(sents, instance['order']))
            random.shuffle(zipped)
            a,b = zip(*zipped)
            data_sentences.append(a)
            data_orders.append(b)
        else:
            data_sentences.append(sents)
            data_orders.append(instance['order'])

    if max_sent_len_ is not None:
        max_sent_len = max_sent_len_
    out_sentences = np.full([len(data_sentences), 5, max_sent_len], vocab['<PAD>'], dtype=np.int32)

    # matrix of sequence lengths
    seq_lens = np.zeros(out_sentences.shape[0:2])

    for i, elem in enumerate(data_sentences):
        for j, sent in enumerate(elem):
            out_sentences[i, j, 0:len(sent)] = sent
            seq_lens[i,j] = len(sent)

    out_orders = np.array(data_orders, dtype=np.int32)

    return out_sentences, out_orders, vocab, seq_lens







