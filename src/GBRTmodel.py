from keras.layers import *
from FeatureGenerator import *
from PointwisePredict import *
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import pickle

class GBRTModel:
    def __init__(self, config, db=None, stats=None, models_as_features=None, load_path=None, w2v=None, base_path="../"):
        if type(config) == str or type(config) == unicode:
            with open(config) as data_file:
                self._config = json.load(data_file)
        else:
            self._config = config

        self.models_as_features = models_as_features

        print "GBRT params:", self._config['hyper_patameters']
        if load_path is None:
            self._model = GradientBoostingClassifier(loss=self._config['hyper_patameters']['loss'],
                                                     learning_rate=self._config['hyper_patameters']['learning_rate'],
                                                     n_estimators=self._config['hyper_patameters']['n_estimators'],
                                                     max_depth=self._config['hyper_patameters']['max_depth'],
                                                     max_features=None)
        else:
            self.loadModel(load_path)

        self._feature_generator = \
            FeatureGenerator(feature_names=self._config['features']['feature_names'],
                             yamada_embedding_path=base_path +
                                                   self._config['features']['yamada_embedding_path'],
#                             yamada_id2title_path=ProjectSettings.getPath()[0] +
#                                                   self._config['features']['yamada_title2id_path'],
                             stats=stats,
                             db=db,
                             models_as_features=self.models_as_features,
                             w2v=w2v)
        self._train_X = []
        self._train_Y = []
        self._db = db

    def getPredictor(self):
        return PointwisePredict(self)

    def predict(self, mention, candidates):
        max_score = -1
        max_entity = None
        ret = dict()
        for candidate in candidates:
            # create feature_vec from mention and candidate and predic prob for pointwise predictor
            feature_vec = self._feature_generator.getFeatureVector(mention, candidate)
            Y = self._model.predict_proba(np.asarray(feature_vec).reshape(1, -1))
            with open('feature_set.txt', 'a') as f:
                f.write('     -> ' + str(Y[0][1]) + '\n')
            ret[candidate] = Y[0][1]
        return ret

    def train(self, mention, candidate, is_correct):
        '''
        Gathers mention and candidate features into a dataFrame
        :param mention:
        :param candidate1: suppose to be None
        :param candidate2: None
        :param correct:
        :return: only builds the _train_df
        '''
        self._train_X.append(self._feature_generator.getFeatureVector(mention, candidate))
        self._train_Y.append(1 if is_correct else 0)

    def is_trainable(self, candidate):
        return True

    def finalize(self):
        '''
        trains the model over accumulated _train_df
        :return:
        '''

        trainX = np.array(self._train_X)
        trainy = np.array(self._train_Y)
        print "fitting gbrt model (", len(self._train_Y), "samples)"
        self._model.fit(trainX, trainy.reshape(trainy.shape[0],))

    def saveModel(self, fname):
        pickle.dump(self._model, open(fname + ".gbrtmodel", "wb"))

    def loadModel(self, fname):
        self._model = pickle.load(open(fname + ".gbrtmodel", "rb"))


