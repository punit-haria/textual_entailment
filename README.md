## Textual Entailment

Suppose we have a story whose sentences are jumbled. Can we build a learning algorithm to correctly reorder the sentences? Such an algorithm will need to understand the directional relation between text fragments. This is known as the problem of predicting textual entailment. 

This repo contains one solution to this problem using recurrent neural networks. At a high-level, the network architecture incorporates layers of [LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) along with a differentiable attention mechanism based on [this paper](https://arxiv.org/abs/1509.06664). The files [rnn_attention.py](./rnn_attention.py) and [rnn_seq_attention.py](./rnn_seq_attention.py) detail two variants of this architecture. 