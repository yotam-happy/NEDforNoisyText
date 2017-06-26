from keras.models import Model
from keras.models import model_from_json
from nltk.corpus import stopwords
import keras.backend as K
import json
import keras.layers as layers
import keras.optimizers as optimizers
import tensorflow as tf
import numpy as np
from Word2vecLoader import DUMMY_KEY
from PointwisePredict import *


def sum_seq(x):
    return K.sum(x, axis=1, keepdims=False)

def to_prob(input):
    sum = K.sum(input, 1, keepdims=True)
    return input / sum

# mask_aware_mean and ZeroMaskedEntries taken from: https://github.com/fchollet/keras/issues/1579

def mask_aware_mean(x):
    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)

    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)

    # compute mask-aware mean of x
    x_mean = K.sum(x, axis=1, keepdims=False) / n

    return x_mean


def my_dot(inputs):
    a = inputs[0] * inputs[1]
    a = K.sum(a, axis=-1, keepdims=True)
    a = K.sigmoid(a)
    return a


def max(x):
    x_max = K.max(x, axis=1, keepdims=False)
    return x_max

class ZeroMaskedEntries(layers.Layer):
    """
    This layer is called after an Embedding layer.
    It zeros out all of the masked-out embeddings.
    It also swallows the mask without passing it on.
    You can change this to default pass-on behavior as follows:

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)
    """

    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, x, mask=None):
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return x * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None


class ModelBuilder:
    def __init__(self, config_json, w2v, entity_transform, categories_transform):
        self._config = config_json
        self._w2v = w2v
        self.entity_transform = entity_transform
        self.categories_transform = categories_transform
        self.optimizer = None

        with tf.device('/cpu:0'):
            self.word_embed_layer = layers.Embedding(self._w2v.wordEmbeddings.shape[0],
                                                     self._w2v.wordEmbeddingsSz,
                                                     input_length=self._config['context_window_size'],
                                                     weights=[self._w2v.wordEmbeddings],
                                                     trainable=self._config['finetune_embd'])
            if entity_transform is None:
                self.concept_embed_layer = layers.Embedding(self._w2v.conceptEmbeddings.shape[0],
                                                            self._w2v.conceptEmbeddingsSz,
                                                            input_length=1,
                                                            weights=[self._w2v.conceptEmbeddings],
                                                            trainable=self._config['finetune_embd'])
            else:
                self.transform_embd_layer = layers.Embedding(entity_transform.get_number_of_values(),
                                                             entity_transform.get_embd_sz(),
                                                             input_length=1,
                                                             weights=[self._w2v.get_random_matrix(entity_transform.get_number_of_values(),
                                                                                                  entity_transform.get_embd_sz())],
                                                             trainable=self._config['finetune_embd'])

            if categories_transform is not None:
                self.categories_embd_layer = layers.Embedding(self._w2v.categoryEmbeddings.shape[0],
                                                              self._w2v.categoryEmbeddingsSz,
                                                              input_length=1,
                                                              weights=[self._w2v.categoryEmbeddings],
                                                              trainable=self._config['finetune_embd'],
                                                              mask_zero=True)

        self.inputs = []
        self.to_join = []
        self.attn = []

    def addCandidateInput(self, name, to_join=True):
        with tf.device('/cpu:0'):
            candidate_input = layers.Input(shape=(1,), dtype='int32', name=name)
            if not self.entity_transform:
                candidate_embed = self.concept_embed_layer(candidate_input)
            else:
                candidate_embed = self.transform_embd_layer(candidate_input)

            candidate_flat = layers.Flatten()(candidate_embed)
        self.inputs.append(candidate_input)
        if to_join:
            self.to_join.append(candidate_flat)
        return candidate_flat

    def addCategoriesInput(self, max_categories, to_join=True):
        with tf.device('/cpu:0'):
            categories_input = layers.Input(shape=(max_categories,), dtype='int32', name='categories_input')
            categories_embed = self.categories_embd_layer(categories_input)
            embed_zero_masked = ZeroMaskedEntries()(categories_embed)
            embed_max = layers.Lambda(max,output_shape=(self._w2v.conceptEmbeddingsSz,))(embed_zero_masked)
        #embed_mean = Lambda(mask_aware_mean)(embed_zero_masked)
        #embed_out = Concatenate([embed_max, embed_mean])

        self.inputs.append(categories_input)
        if to_join:
            self.to_join.append(embed_max)
        return embed_max

    def _buildAttention(self, seq, controller):
        controller_repeated = layers.RepeatVector(self._config['context_window_size'])(controller)
        attention = layers.merge([controller_repeated, seq], mode='concat', concat_axis=-1)
        #attention = layers.concatenate([controller_repeated, seq], axis=-1)

        #layers.Dense(1, activation='sigmoid')
        attention = layers.TimeDistributedDense(1, activation='sigmoid')(attention)
        attention = layers.Flatten()(attention)
        attention = layers.Lambda(to_prob, output_shape=(self._config['context_window_size'],))(attention)

        attention_repeated = layers.RepeatVector(self._w2v.conceptEmbeddingsSz)(attention)
        attention_repeated = layers.Permute((2, 1))(attention_repeated)

        weighted = layers.merge([attention_repeated, seq], mode='mul')
        #weighted = layers.multiply([attention_repeated, seq])
        summed = layers.Lambda(sum_seq, output_shape=(self._w2v.conceptEmbeddingsSz,))(weighted)
        return summed, attention

    def buildAttention(self, seq, controller):
        controller_repeated = layers.RepeatVector(self._config['context_window_size'])(controller)
        controller_repeated = layers.TimeDistributedDense(self._w2v.conceptEmbeddingsSz)(controller_repeated)

        attention = layers.Lambda(my_dot, output_shape=(self._config['context_window_size'],))([controller_repeated, seq])

        attention = layers.Flatten()(attention)
        attention = layers.Lambda(to_prob, output_shape=(self._config['context_window_size'],))(attention)

        attention_repeated = layers.RepeatVector(self._w2v.conceptEmbeddingsSz)(attention)
        attention_repeated = layers.Permute((2, 1))(attention_repeated)

        weighted = layers.merge([attention_repeated, seq], mode='mul')
        summed = layers.Lambda(sum_seq, output_shape=(self._w2v.conceptEmbeddingsSz,))(weighted)
        return summed, attention

    def addContextInput(self, controller=None):
        with tf.device('/cpu:0'):
            left_context_input = layers.Input(shape=(self._config['context_window_size'],), dtype='int32', name='left_context_input')
            right_context_input = layers.Input(shape=(self._config['context_window_size'],), dtype='int32', name='right_context_input')
            self.inputs += [left_context_input, right_context_input]
            left_context_embed = self.word_embed_layer(left_context_input)
            right_context_embed = self.word_embed_layer(right_context_input)

        if self._config['context_network'] == 'gru':
            left_rnn = layers.GRU(self._w2v.wordEmbeddingsSz)(left_context_embed)
            right_rnn = layers.GRU(self._w2v.wordEmbeddingsSz)(right_context_embed)
            self.to_join += [left_rnn, right_rnn]
        elif self._config['context_network'] == 'mean':
            left_mean = layers.Lambda(mask_aware_mean, output_shape=(self._w2v.conceptEmbeddingsSz,))(left_context_embed)
            right_mean = layers.Lambda(mask_aware_mean, output_shape=(self._w2v.conceptEmbeddingsSz,))(right_context_embed)
            self.to_join += [left_mean, right_mean]
        elif self._config['context_network'] == 'attention':
            left_rnn = layers.GRU(self._w2v.wordEmbeddingsSz, return_sequences=True)(left_context_embed)
            right_rnn = layers.GRU(self._w2v.wordEmbeddingsSz, return_sequences=True)(right_context_embed)

            after_attention_left, attn_values_left = \
                self.buildAttention(left_rnn, controller)
            after_attention_right, attn_values_right = \
                self.buildAttention(right_rnn, controller)
            self.to_join += [after_attention_left, after_attention_right]
            self.attn += [attn_values_left, attn_values_right]
        else:
            raise "unknown"

    def addMentionInput(self):
        with tf.device('/cpu:0'):
            mention_input = layers.Input(shape=(self._config['max_mention_words'],), dtype='int32', name='mention_input')
            mention_embed = self.word_embed_layer(mention_input)
            mention_mean = layers.Lambda(mask_aware_mean, output_shape=(self._w2v.conceptEmbeddingsSz,))(mention_embed)
        self.inputs.append(mention_input)
        self.to_join.append(mention_mean)


class DeepModel:
    def __init__(self, config, load_path=None, w2v=None, db=None,
                 stats=None, models_as_features=None, entity_transform=None, inplace_transform=False,
                 categories_transform=None):
        '''
        Creates a new NN model configured by a json.

        config is either a dict or a path to a json file

        json structure:
        {
            strip_stop_words=[boolean]
            context_window_size=[int]
            max_mention_words=[int]
            dropout=[0.0 .. 1.0]
            feature_generator={mention_features={feature names...}, entity_features={feature names...}}

            finetune_embd=[boolean]
            pairwise=[boolean]
            inputs=[list out of ['candidates', 'context', 'mention']]
        }
        '''

        if type(config) in {unicode, str}:
            with open(config) as data_file:
                self._config = json.load(data_file)
        else:
            self._config = config
        self._stopwords = stopwords.words('english') if self._config['strip_stop_words'] else None
        self._special_tokens = self._config['special_tokens'] if 'special_tokens' in self._config else False

        self._word_dict = None
        self._concept_dict = None

        self.entity_transform = entity_transform
        self.inplace_transform = inplace_transform
        self.categories_transform = categories_transform
        self.max_categories = self._config['max_categories'] if 'max_categories' in self._config else None
        self._db = db
        self._batch_left_X = []
        self._batch_right_X = []
        self._batch_categories_X = []
        self._batch_candidate_X = []
        self._batch_mention_X = []
        self._batchY = []
        self._last_layer_sz = 0
        self.train_loss = []
        self._batch_size = 128
        self.inputs = {x for x in self._config['inputs']}
        self._w2v = w2v

        self.model = None
        self.get_attn_model = None

        if load_path is None:
            self.compileModel(w2v)
        else:
            self.compileModel(w2v, load_path)

    def getPredictor(self):
        return PointwisePredict(self)

    def compileModel(self, w2v, weights_file=None):
        self._word_dict = w2v.wordDict
        self._concept_dict = w2v.conceptDict

        model_builder = ModelBuilder(self._config, w2v, self.entity_transform, self.categories_transform)

        attn_controller = None
        if 'candidates' in self.inputs:
            attn_controller = model_builder.addCandidateInput('candidate1_input', to_join='candidates' in self.inputs)

        if 'categories' in self.inputs:
            c = model_builder.addCategoriesInput(max_categories=self.max_categories)
            if attn_controller == None:
                attn_controller = c
            else:
                attn_controller = layers.merge([attn_controller, c], mode='concat', concat_axis=-1)
                #attn_controller = layers.concatenate([attn_controller, c])

        if 'context' in self.inputs:
            model_builder.addContextInput(controller=attn_controller)

        if 'mention' in self.inputs:
            model_builder.addMentionInput()

        inputs = model_builder.inputs
        to_join = model_builder.to_join
        attn = model_builder.attn

        # join all inputs
        #x = layers.concatenate(to_join) if len(to_join) > 1 else to_join[0]
        x = layers.merge(to_join, mode='concat') if len(to_join) > 1 else to_join[0]

        # build classifier model
        for c in self._config['classifier_layers']:
            x = layers.Dense(c, activation='relu')(x)
        if 'dropout' in self._config:
            x = layers.Dropout(float(self._config['dropout']))(x)
        out = layers.Dense(2, activation='softmax', name='main_output')(x)

        self.model = Model(input=inputs, output=[out])
        #self.model = Model(inputs=inputs, outputs=[out])
        if weights_file is not None:
            self.model.load_weights(weights_file + ".weights")
#optimizers.TFOptimizer( )
        self.model.compile(optimizer=tf.train.AdagradOptimizer(0.1), loss='binary_crossentropy')
        self.get_attn_model = Model(input=inputs, output=attn)
        print "model compiled!"

    def is_trainable(self, candidate):
        if candidate is not None and self.inplace_transform:
            candidate = self.entity_transform.entity_id_transform(candidate)

        if candidate is None or (self.entity_transform is None and candidate not in self._concept_dict):
            return False
        return True

    def _2vec(self, mention, candidate):
        """
        Transforms input to w2v vectors
        returns a tuple: (wikilink vec, candidate1 vec, candidate2 vec)

        if cannot produce wikilink vec or vectors for both candidates then returns None
        if cannot produce vector to only one of the candidates then returns the id of the other
        """

        if not self.is_trainable(candidate):
            return None

        if candidate is not None and self.inplace_transform:
            candidate = self.entity_transform.entity_id_transform(candidate)

        candidate_X = None
        categories_X = None
        left_context_X = None
        right_context_X = None
        mention_X = None

        # get candidate inputs
        if 'candidates' in self.inputs:
            candidate_X = np.array([self._concept_dict[candidate] if self.entity_transform is None else candidate])


        # get context input
        if 'context' in self.inputs:
            left_context_X = self.wordIteratorToIndices(mention.left_context_iter(),
                                                        self._config['context_window_size'])
            right_context_X = self.wordIteratorToIndices(mention.right_context_iter(),
                                                         self._config['context_window_size'])

        # get mention input
        if 'mention' in self.inputs:
            mention_X = self.wordIteratorToIndices(mention.mention_text_tokenized(),
                                                   self._config['max_mention_words'])

        if 'categories' in self.inputs:
            categories_X = [c for c in self._w2v.page_categories[candidate]] if candidate in self._w2v.page_categories else []
            if len(categories_X) == 0:
                categories_X.append(self._w2v.page_categories['~Empty~'])
            if len(categories_X) > self.max_categories:
                categories_X = categories_X[:self.max_categories]
            while len(categories_X) < self.max_categories:
                categories_X.append(self._w2v.page_categories['~Dummy~'])
            categories_X = np.array(categories_X)

        return left_context_X, right_context_X, mention_X, candidate_X, categories_X

    def wordIteratorToIndices(self, it, output_len):
        o = []
        for w in it:
            w = w.lower()
            if len(o) >= output_len:
                break
            if w in self._word_dict:
                if self._stopwords is not None and w in self._stopwords:
                    continue
                o.append(self._word_dict[w])
                continue
        if len(o) == 0:
            o.append(self._word_dict[DUMMY_KEY])

        o = o[:: -1]
        arr = np.zeros((output_len,))
        n = len(o) if len(o) <= output_len else output_len
        arr[:n] = np.array(o)[:n]
        return arr

    def get_context_indices(self, it, output_len):
        words = []
        indices = []
        for i, w in enumerate(it):
            w = w.lower()
            words.append(w)
            if len(indices) >= output_len:
                break
            if w in self._word_dict:
                if self._stopwords is not None and w in self._stopwords:
                    continue
                indices.append(i)
        return words, indices

    def train(self, mention, candidate, is_correct):
        """
        Takes a single example to train
        :param mention:    The mention to train on
        :param candidate:  the candidate entity id
        :param is_correct:
        """
        vecs = self._2vec(mention, candidate)
        if not isinstance(vecs, tuple):
            return # nothing to train on

        (left_X, right_X, mention_X, candidate_X, categories_X) = vecs
        Y = np.array([1, 0] if is_correct else [0, 1])
        self._trainXY(left_X, right_X, mention_X, candidate_X, categories_X, Y)

    def _trainXY(self, left_X, right_X, mention_X, candidate_X, categories_X, Y):
        self._batch_left_X.append(left_X)
        self._batch_right_X.append(right_X)
        self._batch_mention_X.append(mention_X)
        self._batch_candidate_X.append(candidate_X)
        self._batch_categories_X.append(categories_X)
        self._batchY.append(Y)

        if len(self._batchY) >= self._batch_size:
            batchX = {}
            if 'candidates' in self.inputs:
                batchX['candidate1_input'] = np.array(self._batch_candidate_X)
            if 'context' in self.inputs:
                batchX['left_context_input'] = np.array(self._batch_left_X)
                batchX['right_context_input'] = np.array(self._batch_right_X)
            if 'mention' in self.inputs:
                batchX['mention_input'] = np.array(self._batch_mention_X)
            if 'categories' in self.inputs:
                batchX['categories_input'] = np.array(self._batch_categories_X)
            batchY = np.array(self._batchY)

            if 'special' not in self._config:
                loss = self.model.train_on_batch(batchX, batchY)
            else:
                jj = self._last_layer_sz
                batchYAll = {}
                while jj > 1:
                    batchYAll['main_output' + jj] = batchY
                    jj /= 2
                loss = self.model.train_on_batch(batchX, batchYAll)

            self.train_loss.append(loss)

            self._batch_left_X = []
            self._batch_right_X = []
            self._batch_mention_X = []
            self._batch_candidate_X = []
            self._batch_categories_X = []
            self._batchY = []

    def finalize(self):
        pass

    def saveModel(self, fname):
        with open(fname+".model", 'w') as model_file:
            model_file.write(self.model.to_json())
        self.model.save_weights(fname + ".weights", overwrite=True)

        with open(fname+".w2v.def", 'w') as f:
            f.write(json.dumps(self._word_dict)+'\n')
            f.write(json.dumps(self._concept_dict)+'\n')

    def loadModel(self, fname):
        with open(fname+".model", 'r') as model_file:
            self.model = model_from_json(model_file.read(), {"ZeroMaskedEntries": ZeroMaskedEntries})
        self.model.load_weights(fname + ".weights")

        with open(fname+".w2v.def", 'r') as f:
            l = f.readlines()
            self._word_dict = {str(x): int(y) for x,y in json.loads(l[0]).iteritems()}
            self._concept_dict = {int(x) if str(x) != DUMMY_KEY else DUMMY_KEY: int(y) for x, y in json.loads(l[1]).iteritems()}

        self.model.compile(optimizer=tf.train.AdagradOptimizer(0.1), loss='binary_crossentropy')

    def predict(self, mention, candidates):
        _batch_left_X = []
        _batch_right_X = []
        _batch_mention_X = []
        _batch_candidate_X = []
        _batch_categories_X = []

        actual_candidates = []
        for candidate in candidates:
            vecs = self._2vec(mention, candidate)
            if not isinstance(vecs, tuple):
                continue
            (left_X, right_X, mention_X, candidate_X, categories_X) = vecs
            _batch_left_X.append(left_X)
            _batch_right_X.append(right_X)
            _batch_mention_X.append(mention_X)
            _batch_candidate_X.append(candidate_X)
            _batch_categories_X.append(categories_X)
            actual_candidates.append(candidate)

        if len(actual_candidates) == 0:
            return None

        batchX = {}
        if 'candidates' in self.inputs:
            batchX['candidate1_input'] = np.array(_batch_candidate_X)
        if 'context' in self.inputs:
            batchX['left_context_input'] = np.array(_batch_left_X)
            batchX['right_context_input'] = np.array(_batch_right_X)
        if 'mention' in self.inputs:
            batchX['mention_input'] = np.array(_batch_mention_X)
        if 'categories' in self.inputs:
            batchX['categories_input'] = np.array(_batch_categories_X)

        batchY = self.model.predict(batchX, batch_size=len(actual_candidates))
        Y = {}
        for i, candidate in enumerate(actual_candidates):
            Y[candidate] = batchY[i][0]
        return Y

    def get_attn(self, mention, candidate):
        vecs = self._2vec(mention, candidate)
        if not isinstance(vecs, tuple):
            return None
        (left_X, right_X, mention_X, candidate_X, categories_X) = vecs

        X = {}
        if 'candidates' in self.inputs:
            X['candidate_input'] = candidate_X.reshape((1, candidate_X.shape[0],))
        if 'context' in self.inputs:
            X['left_context_input'] = left_X.reshape((1, left_X.shape[0],))
            X['right_context_input'] = right_X.reshape((1, right_X.shape[0],))
        if 'mention' in self.inputs:
            X['mention_input'] = mention_X.reshape((1, mention_X.shape[0],))
        if 'categories' in self.inputs:
            X['categories_input'] = categories_X.reshape((1, categories_X.shape[0],))

        attn_out = self.get_attn_model.predict(X, batch_size=1)

        left_context, left_indices = self.get_context_indices(mention.left_context_iter(),
                                                              self._config['context_window_size'])
        right_context, right_indices = self.get_context_indices(mention.right_context_iter(),
                                                                self._config['context_window_size'])
        left_attn = [0 for i in xrange(len(left_context))]
        right_attn = [0 for i in xrange(len(right_context))]
        for i in xrange(self._config['context_window_size']):
            if i < len(left_indices):
                left_attn[left_indices[i]] = attn_out[0][0, i]
            if i < len(right_indices):
                right_attn[right_indices[i]] = attn_out[1][0, i]
        return left_context, left_attn, right_context, right_attn
