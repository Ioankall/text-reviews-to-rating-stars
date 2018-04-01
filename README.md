# text_review_to_rating_stars_transformer

Using Machine Learning techniques (primarily deep learning), we will create a model that provided a text input (eg. a review for a restaurant) will give us a rating in the form of [1-5] stars.

In order to train our model we use the dataset provided from Yelp containing reviews for venues. The json files provided from Yelp have been imported in a Mongo DB, so our scripts will get their train data from Mongo.

At the preprocessing stage, we remove stopwords, punctuation and numerical digits. Then we tokenize the text and create a vocabulary with the 80.000 most frequent words. Each word is mapped to an integer. Finally, in each review we replace each word with its mapped integer.

As far as it concerns our models we created two different ones. We set up two Neural Networks: a CNN and an LSTM. The first one reached an accuracy of 60.5% while the second one made it to 62.4%. Also, on training our model we used the GloVe embeddings with a dimension of 200.

There are a lot of parameters that we can experiment with and they may lead us to an even better result.

Fianlly, there is a server.py file that starts a RESTful API. Its purpose is to provide a call that will receive text as a parameter and will return the rating. We use this for demonstrating the results.
