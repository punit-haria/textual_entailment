## Textual Entailment

Building machines with commonsense reasoning is a really important and interesting research problem. Given two sentences, the problem of textual entailment is the problem of deciding whether the first sentence can be used to prove that the second is true, false, or uncorrelated. This ability is very useful in many applications including question-answering and text summarization. 

One interesting question we may ask is whether an agent can learn to make inferences and deductions from textual stories. Suppose we're given a short narrative which has been jumbled, that is, the sentences have been randomly shuffled. Can we build a learning algorithm to correctly reorder these sentences? This will require both commonsense knowledge and temporal understanding.

This repo contains one solution to this problem using recurrent neural networks. At a high-level, the network architecture incorporates layers of [LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) along with a differentiable attention mechanism based on [this paper](https://arxiv.org/abs/1509.06664). The files [rnn_attention.py](./rnn_attention.py) and [rnn_seq_attention.py](./rnn_seq_attention.py) detail two variants of this architecture. 


