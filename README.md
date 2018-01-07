# movie-riview-
This is done by the data provided by the corenell which consists of 1403 movie riviews  in it and , I used an LSTM model to get the rating of the movie,LSTM is a recurrent neural network.

## Requirements:
1. Anaconda - This installs python along with most popular python libraries including sklearn. If not already installed, install it from https://www.continuum.io/downloads . 
2. Keras- Python library for Deep laerning.
3. Glove Model - Google pre trained word vecetor representation based on the news data set.

## Dataset Preparation :
1.  Run the code `lstm.py` then the `.npy` files will be created which has the word vector pairing of all the words in the corpus

## Output Format:
1. The LSTM model will be compiled and fitted into the data we give in and the model will be stored in the `.h5` formaat for the furthur use
2.Now the output will be of the format softmax vector of the probabilities of the rating.
