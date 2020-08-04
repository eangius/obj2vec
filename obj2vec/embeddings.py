# Third party libraries
import csv
import hashlib
import numpy as np
import os
import pandas as pd
import random

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import dot, Dense, Embedding, Input, Reshape
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


class Obj2Vec:

    # CONSTRUCTION METHODS #

    # Instantiate an object-to-vector model either for training or usage.
    #
    # @param filename:   optional path to pre-trained embedding *.tsv.zip file.
    # @param vocabulary: optional list of object keys that must:
    #                    -- uniquely encode distinct objects.
    #                    -- index[0] encode the unknown object.
    def __init__(self, filename: str = None, vocabulary: list = None):
        require(bool(vocabulary) != bool(filename),
                "need a vocabulary or pre-trained embedding")

        if bool(filename):
            require(os.path.exists(filename), "Embedding file not found")
            self.embedding = pd.read_csv(
                filename,
                sep='\t',
                compression='gzip',
                index_col=[self.index_col],
                na_filter=False,
                quoting=csv.QUOTE_NONE,
            )

        if bool(vocabulary):
            require(
                len(vocabulary) == len(set(vocabulary)),
                "vocabulary cannot have duplicates"
            )
            self.embedding = pd.DataFrame(
                vocabulary,
                columns=[self.index_col],
                dtype=str
            ).set_index(self.index_col)

        self.architecture = None
        self.history = None

    # Compiles a skip-gram with negative-sampling architecture. This arch trains
    # separate embeddings for target & context objects that get compared by
    # their cosine similarity to internally classify if targets are within or
    # outside context window. Only target embedding is kept as vectors.
    #
    # @param embed_dim:  optional number of "concepts" to capture.
    # @param learn_rate: fidelity of learning vs speed.
    # @return self
    def build(self, embed_dim: int = None, learn_rate: float = 0.01):
        require(learn_rate > 0, "positive learn rate needed")

        dim = self._dimensionality()
        if not embed_dim:
            embed_dim = dim

        require(embed_dim > 0, "positive dimension needed")

        if embed_dim < dim:
            self.embedding = self._update_embedding(
                PCA(n_components=embed_dim).fit_transform(self.embedding)
            )

        self.architecture = self._build_sgns(
            vocab_size=self.embedding.shape[0],
            embed_dim=embed_dim,
            weights=None if self.embedding.empty else self.embedding.to_numpy()
        )
        self.architecture.compile(
            optimizer=SGD(lr=learn_rate),  # best qual & speed
            loss='binary_crossentropy',
            metrics=[
                Precision(name='%p'),
                Recall(name='%r')
            ]
        )
        return self

    # Trains or continue training the architecture with specific skip-gram data.
    # Formulation of skip-gram data depends on model use-case but should be a
    # list of tuples: 1st item are lists of target/context object keys (not
    # not indices) & 2nd item is list of labels. Unrecognized objects in are
    # kept as unknown vocabulary values.
    #
    # @param skip_grams: data set to learn from.
    # @param epochs:     number of re-iterations of data.
    # @param eval_data:  fraction of data to use for evaluation.
    # @param batch_size: chunk of examples to process while learning.
    # @return self
    def learn(self, skip_grams: list, epochs: int = 1, eval_data: float = 0.0,
              batch_size: int = 128):
        require(len(skip_grams) > 0, "non-empty skip-gram data needed to learn")
        require(epochs > 0, "positive epochs needed")
        require(0 <= eval_data <= 1, "evaluation data fraction is out of range")
        require(self.architecture, "must first build the architecture")

        targets, contexts, labels = self._parse_skipgrams(skip_grams)
        self.history = self.architecture.fit(
            x=[
                np.array(targets, dtype='int32'),
                np.array(contexts, dtype='int32')
            ],
            y=np.array(labels, dtype='int32'),
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=eval_data,
            shuffle=True,
            callbacks=[
                ModelCheckpoint(
                    filepath='model.h5',
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=2
                )
            ],
        )

        self.embedding = self._update_embedding(
            self.architecture.get_layer(name='target_embed').get_weights()[0]
        )
        return self

    # Evaluates model for production purposes.
    #
    # @param skip_grams: data set to learn from.
    # @return quality:   keras quality object.
    def evaluate(self, skip_grams: list):
        require(self.architecture, "this embedding needs learning")

        targets, contexts, labels = self._parse_skipgrams(skip_grams)
        return self.architecture.evaluate(
            x=[
                np.array(targets, dtype='int32'),
                np.array(contexts, dtype='int32')
            ],
            y=np.array(labels, dtype='int32'),
            verbose=1
        )

    # Concatenates another trained embedding with the same vocabulary onto
    # this one. Useful to enrich vectors with different forms of context.
    #
    # @param other: other object-to-vector
    # @return self
    def append(self, other):
        require(not self.embedding.empty, "this embedding needs learning")
        require(not other.embedding.empty, "other embedding needs learning")
        require(self.embedding.index.equals(other.embedding.index),
                "object vocabularies differ")

        columns = range(
            len(self.embedding.columns) + len(other.embedding.columns)
        )
        self.embedding = pd.concat([self.embedding, other.embedding], axis=1)
        self.embedding.columns = list(columns)
        return self

    # Shrinks rows & columns of an embedding as a way to reduce the memory
    # footprint & noise. Note however that this transformation is irreversible
    # & may affect embedding quality.
    #
    # @return vocabulary    optional subset of objects to keep.
    # @return dim_variance  optional fraction of dimension signal to keep.
    # @return self
    def compact(self, vocabulary: list = None, dim_variance: float = None):
        if bool(vocabulary):
            self.embedding = self.embedding.drop(
                set(self.embedding.index) - set(vocabulary),
                errors='ignore'
            )

        if bool(dim_variance):
            require(0 < dim_variance <= 1, "fraction of signal to keep")
            self.embedding = self._update_embedding(
                PCA(n_components=dim_variance).fit_transform(self.embedding)
            )

        return self

    # Serializes embedding as a compressed TSV archive.
    def save(self, filename: str):
        self.embedding.to_csv(
            filename,
            sep='\t',
            compression='gzip',
            index=True,
        )
        return self

    # USAGE METHODS #

    # Looks up an embedding vector for the object. Out of vocabulary objects
    # produce a unique & deterministic random permutation of the UNKNOWN vector.
    #
    # @param obj:   object key.
    # @return vtr
    def vectorize(self, obj: str) -> pd.Series:
        require(not self.embedding.empty, "this embedding needs learning")

        if obj in self.embedding.index:
            vtr = pd.Series.copy(self.embedding.loc[obj], deep=True)
        else:
            num = int(hashlib.md5(str(obj).encode('utf-8')).hexdigest(), 16)
            idx = str(num % self._dimensionality())

            random.seed(num)
            vtr = pd.Series.copy(self._unknown_vtr(), deep=True).rename(obj)
            unique = False
            while not unique:
                vtr.at[idx] = random.uniform(-1, 1)
                unique = not self._exists_mask(vtr).any()
        return vtr

    # Inverse of vectorize to find the closes object from a possibly non
    # existing vector.
    #
    # @param vtr:   vector.
    # @return obj:
    def objectify(self, vtr: pd.Series) -> str:
        require(not self.embedding.empty, "must first load an embedding")

        keep = self._exists_mask(vtr)
        if self.embedding[keep].shape[0] == 1:
            result = self.embedding[keep].index
        else:
            result = self.rank(vtr, n=1)
        return result[0]

    # Compares cosine similarity between vectors.
    #
    # @param obj1:  one object.
    # @param obj2:  another object.
    # @return sim:  1 = synonyms, 0 = irrelevant, -1 = antonyms
    def similarity(self, obj1: str, obj2: str) -> float:
        vtr1 = self.vectorize(obj1).to_numpy()
        vtr2 = self.vectorize(obj2).to_numpy()
        return np.dot(vtr1, vtr2)/(np.linalg.norm(vtr1) * np.linalg.norm(vtr2))

    # Finds top most similar vectors listed in descending relevance order.
    #
    # @param objvtr:    object key or vector.
    # @param n:         number of results to retrieve.
    # @return           list of nearest objects.
    def rank(self, objvtr, n: int = 10) -> list:
        require(n > 0, "need number of results to fetch")

        if type(objvtr) == pd.Series:
            vtr = objvtr
            obj = vtr.name if vtr.name else self._unknown_vtr().name
        else:
            vtr = self.vectorize(objvtr)
            obj = objvtr

        # cluster by (one extra) nearest neighbor.
        finder = NearestNeighbors(
            radius=0.2,
            n_jobs=-1,
        ).fit(self.embedding)
        hits = finder.kneighbors(
            [vtr],
            n_neighbors=n+1,
            return_distance=False
        )[0]

        # convert indices to objects & filter self.
        result = list()
        for i in hits:
            obj_other = self.embedding.iloc[i].name
            if obj != obj_other:
                result.append(obj_other)

        # order by our similarity
        return sorted(
            result,
            reverse=True,
            key=lambda obj_other: self.similarity(obj, obj_other)
        )

    # PRIVATE METHODS #

    @staticmethod
    def _build_sgns(vocab_size: int, embed_dim: int,
                    weights: np.ndarray = None):
        require(embed_dim < vocab_size,
                "dimension cannot be larger than vocabulary")

        shape = (1,)
        input_target = Input(shape, name='target_obj')
        input_context = Input(shape, name='context_obj')

        def build_embedding(input: Input, name: str):
            lyr = Embedding(
                input_dim=vocab_size,
                output_dim=embed_dim,
                input_length=1,
                weights=None if weights is None else [weights],
                name=f'{name}_embed',
            )(input)
            lyr = Reshape((embed_dim, 1), name=f'{name}_reshape')(lyr)
            return lyr

        dot_product = Reshape(shape, name=f'cosine_reshape')(dot([
            build_embedding(input_target, name='target'),
            build_embedding(input_context, name='context'),
        ], axes=1, normalize=True, name='cosine_sim'))
        result = Dense(1, activation=sigmoid, name='classifier')(dot_product)

        return Model(
            inputs=[input_target, input_context],
            outputs=result,
            name='skip_gram_negative_sample'
        )

    # Splits skip-grams into array of its parts. Reverse maps objects
    # to embedding's vocabulary indices filtering any unknown gram.
    def _parse_skipgrams(self, skip_grams: list) -> (list, list, list):
        targets, contexts, labels = [], [], []
        for i, (pairs, lbls) in enumerate(skip_grams):
            terms = list(zip(*pairs))
            targets.extend(self._index_of(obj) for obj in terms[0])
            contexts.extend(self._index_of(obj) for obj in terms[1])
            labels.extend(lbls)
        return targets, contexts, labels

    # Dataframe mask matching to row of values.
    def _exists_mask(self, vtr: pd.Series):
        require(vtr.shape == (self._dimensionality(),),
                "vector needs same dimension as embedding")
        keep = 1
        for col in self.embedding.columns:
            keep = keep & (self.embedding[col] == vtr[col])
        return keep

    def _index_of(self, obj: str):
        if obj in self.embedding.index:
            return self.embedding.index.get_loc(obj)
        else:
            return 0  # unknown

    def _dimensionality(self):
        return self.embedding.shape[1] if not self.embedding.empty else 0

    def _unknown_vtr(self):
        return self.embedding.iloc[0]

    def _update_embedding(self, data):
        df_embed = pd.DataFrame(data)  # idx->vtr
        df_terms = self.embedding \
            .drop(self.embedding.columns, axis=1).reset_index()  # idx->obj
        return df_terms.merge(
            df_embed,
            left_index=True,
            right_index=True,
        ).set_index(self.index_col)  # obj->vtr

    # MEMBER VARIABLES #

    index_col = 'object'


# Assert like helper to validate inputs at runtime.
def require(condition: bool, msg: str = ""):
    if not condition:
        raise ValueError(msg)
    return
