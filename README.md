# PyTorch implementation of LRU (Linear Recurrent Unit)

This Repository implements a pytorch version of the LRU, including parallel training and serial inference. 

The parallel acceleration is achieved by Upper/Lower algorithm based on [Bojbojone/rnn](https://github.com/bojone/rnn).

# About LRU

LRU is proposed by [Resurrecting Recurrent Neural Networks](https://arxiv.org/abs/2303.06349) for Long Sequences as an alternative to RNNs, with an architecture inspired by SSMs. I conducted experiments on a classification problem in finance, and the results showed a significant improvement of LRU over LSTM. 

## Notes

If you want to use LRU for time series prediction, please note that the paper requires LRU with MLP for this, while merely LRU is implemented in this project.
