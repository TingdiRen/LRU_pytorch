# PyTorch implementation of LRU (Linear Recurrent Unit)

This Repository implements a pytorch version of the LRU, including parallel training and serial inference. 

The parallel acceleration is achieved by Upper/Lower algorithm based on [Bojbojone/rnn](https://github.com/bojone/rnn).

## Performance

I conducted experiments on a classification problem in finance, and the results showed a significant improvement of LRU over LSTM. 

## Notes

If you want to use LRU for time series prediction, please note that the paper requires LRU with MLP for this, while merely LRU is implemented in this project.