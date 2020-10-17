
# Neural Word Embedding
In this assignment we have followed [this](https://github.com/bentrevett/pytorch-sentiment-analysis) awesome github repo by <b>Ben Trevett</b> on sentiment analysis in pytorch.
He has taken us through the following 6 different models trained on IMDB reviews for positive and negative sentiments. We have also tried to replicate the same 6 codes.

## 1. Simple Sentiment Analysis
Trained a Simple RNN model using TorchText library

Our trial : https://github.com/gdeotale/E4P2/blob/master/Assignment9/Neural_word_embeddings_and_sentiment_analysis_ex1.ipynb 

## 2. Better Sentiment Analysis
Improved previous model using -
  - packed padded sequences
  - pre-trained word embeddings
  - different RNN architecture
  - bidirectional RNN
  - multi-layer RNN
  - regularization(dropout)
  - different optimizer - Adam

Our trial : https://github.com/gdeotale/E4P2/blob/master/Assignment9/Neural_word_embeddings_and_sentiment_analysis_ex2.ipynb

Results: 
![](https://github.com/gdeotale/E4P2/blob/master/Assignment9/model_2_predictions.PNG)

## 3. Faster Sentiment Analysis
