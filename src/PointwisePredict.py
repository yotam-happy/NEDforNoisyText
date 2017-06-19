import operator
from matplotlib import pyplot as plt
import numpy as np

class PointwisePredict:
    """
    This model takes a pairwise model that can train/predict on pairs of candidates for a wikilink
    and uses it to train/predict on a list candidates using a knockout method.
    The candidates are taken from a stats object
    """

    def __init__(self, pointwise_model):
        """
        :param pairwise_model:  The pairwise model used to do prediction/training on a triplet
                                (wikilink,candidate1,candidate2)
        :param stats:           A statistics object used to get list of candidates
        """
        self._pointwise_model = pointwise_model
        self.predicts = []

    def predict(self, mention):
        if len(mention.candidates) < 1:
            return None

        d = self._pointwise_model.predict(mention, mention.candidates)
        if d is None:
            return d
        d = {candidate: score for candidate, score in d.iteritems() if score is not None}
        if len(d) == 0:
            return d
        return max(d.iteritems(), key=operator.itemgetter(1))[0]

    def predict_prob(self, mention, candidate):
        a = self._pointwise_model.predict(mention, candidate)
        if a is not None:
            self.predicts.append(a)
        return a if a is not None else 0

    def plot_predict_hist(self, step=0.1):
        bins = np.arange(0.0, 1.0, step)
        plt.xlim([min(self.predicts)-step, max(self.predicts)+step])
        plt.hist(self.predicts, bins=bins, alpha=0.5)
        plt.title('Prediction values distribution')
        plt.xlabel(- 7)
