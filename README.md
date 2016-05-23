This code is a substantial modification of Yoon Kim's torch implementation of Word2Vec. 

This version includes:

* subsampling, as per the original word2vec C code;
* minibatch processing
* the ability to load an existing semantic space and train it further; 
* a function to print out the new semantic space (compatible with the DISSECT .dm format).

The arguments to the program are:

* -corpus: the corpus to be processed
* -background: the background space to be loaded (default: none)
* -window: the window size on either size of the target word (+/- n words, default 2)
* -minfreq: the minimum frequency a token has to have to be considered (default 1)
* -dim: the dimension of the embeddings (default 100)
* -lr: the learning rate (default 0.5)
* -min_lr: the minimum learning rate (default 0.001)
* -neg_samples: the number of negative samples (default 5)
* -table_size: size of the table for negative sampling (default 1e8)
* -epochs: number of epochs (default 1)
* -gpu: whether to use gpu or not (default 0)
* -stream: whether to stream data from the hard drive or pre-load it (default 1. **In its current implementation, the code only works with streaming.**)
* -batch_size: minibatch size (default 1)
* -subsampl: subsampling term (default 0.0001. Reduce for small corpora: e.g. 0.001 for book-length corpus)
* -alpha: smoothing term for word frequencies (default 0.75)

Example usage:

th main.lua -corpus wikipedia.sentences -minfreq 5 -epochs 5 -batch_size 10 > wikipedia.dm 
