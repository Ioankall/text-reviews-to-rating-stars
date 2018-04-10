import preprocessing
import yelp_neural_networks
import svm_classification as svm

from pymongo import MongoClient
from itertools import chain

from sklearn.metrics import confusion_matrix


def error_cost(y_pred, y_val):
    cost = 0
    for i in range(0, len(y_pred)):
        if abs(y_pred[i] - y_val[i]) > 1:
            cost = cost + abs(y_pred[i] - y_val[i])
    cost = (1.0 * cost) / len(y_pred)
    return cost


if __name__ == '__main__':

    client = MongoClient('localhost', 27017)
    db = client.Yelp

    reviews_1 = db.reviews.find({'stars': 1}, {'text': 1, 'stars': 1}).limit(500000)
    reviews_2 = db.reviews.find({'stars': 2}, {'text': 1, 'stars': 1}).limit(500000)
    reviews_3 = db.reviews.find({'stars': 3}, {'text': 1, 'stars': 1}).limit(500000)
    reviews_4 = db.reviews.find({'stars': 4}, {'text': 1, 'stars': 1}).limit(500000)
    reviews_5 = db.reviews.find({'stars': 5}, {'text': 1, 'stars': 1}).limit(500000)
    reviews = chain(reviews_1, reviews_2, reviews_3, reviews_4, reviews_5)

    result, stars = preprocessing.process_data(reviews, lexicon='save')

    svm.train_model(result, stars)
    # yelp_neural_networks.train_model('lstm', result, stars)
    # yelp_neural_networks.train_model('cnn', result, stars)

    # svm.evaluate_model(result, stars)
    # yelp_neural_networks.evaluate_model('lstm', result, stars)
    # yelp_neural_networks.evaluate_model('cnn', result, stars)

    # predictions = yelp_neural_networks.predict_model('lstm', result)

    # print error_cost(predictions, stars)
    # print confusion_matrix(stars, predictions, labels=[1, 2, 3, 4, 5])


