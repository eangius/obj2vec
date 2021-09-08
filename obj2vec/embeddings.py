#!usr/bin/env python

# External libraries
import csv
import hashlib
import numpy as np
import os
import pandas as pd
import random
from cachetools import LRUCache, cached
from collections import defaultdict
from typing import Tuple
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import dot, Dense, Embedding, Input, Reshape
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


class Obj2Vec:

    # CONSTRUCTION METHODS #

    # Instantiate an object-to-vector model either for training or usage.
    # @param filename:   optional path to pre-trained embedding *.tsv.xz file.
    # @param vocabulary: optional list of object keys that must:
    #                    -- uniquely encode distinct objects.
    #                    -- index[0] encode the unknown object.
    def __init__(self, filename: str = None, vocabulary: list = None):
        require(bool(vocabulary) != bool(filename),
                "need a vocabulary or pre-trained embedding")

        if filename:
            # TODO: handle loading very large embeddings.
            require(os.path.exists(filename), "Embedding file not found")

            col_schema = defaultdict(lambda: self.resolution)
            col_schema[0] = 'str'

            self.embedding = pd.read_csv(
                filename,
                sep='\t',
                compression=self.compression_format,
                header=None,
                index_col=[0],
                na_filter=False,        # else corruption
                dtype=col_schema,       # faster loading
                quoting=csv.QUOTE_NONE,
            )
            self.embedding.rename(
                columns={c: c - 1 for c in self.embedding.columns},  # zero col start
                inplace=True,
            )

        if vocabulary:
            require(
                len(vocabulary) == len(set(vocabulary)),
                "vocabulary cannot have duplicates"
            )
            self.embedding = pd.DataFrame(
                vocabulary,
                columns=[0],
                dtype=str
            ).set_index(0)

        self.embedding.index.rename(name=self.index_col, inplace=True)
        self.architecture = None
        self.history = None
        return

    # Compiles a skip-gram with negative-sampling architecture. This arch trains
    # separate embeddings for target & context objects that get compared by
    # their cosine similarity to internally classify if targets are within or
    # outside context window. Only target embedding is kept as vectors.
    # @param embed_dim:  optional number of "concepts" to capture.
    # @param learn_rate: fidelity of learning vs speed.
    # @return self
    def build(self, embed_dim: int = None, learn_rate: float = 0.01):
        require(learn_rate > 0, "positive learn rate needed")

        dim = self._dimensionality()
        embed_dim = embed_dim or dim

        if embed_dim < dim:
            self.compact(embed_dim=embed_dim)

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
    # not indices) & 2nd item is list of labels. Unrecognized objects are
    # kept as unknown vocabulary values.
    # @param skip_grams: data set to learn from.
    # @param epochs:     number of re-iterations of data.
    # @param eval_data:  fraction of data to use for evaluation.
    # @param batch_size: chunk of examples to process while learning.
    # @return self
    def learn(self, skip_grams: list, epochs: int = 1, eval_data: float = 0.0, batch_size: int = 128):
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
                    filepath=self.checkpoint_file,
                ),
                EarlyStopping(
                    monitor='loss',
                    patience=2
                )
            ],
        )

        self.embedding = self._update_embedding(
            self.architecture.get_layer(name='target_embed').get_weights()[0]
        )
        return self

    # Evaluates model for production purposes.
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

    # Adds new objects in the embedding. Note that the new vectors will be bogus
    # until the embedding is re-trained. Useful for extending off-the-shelf
    # with custom vocabularies.
    # @param vocabulary: set of objects to add or modify
    # @return self
    def extend(self, vocabulary: set):

        # don't overwrite existing or unknown vector terms
        new_terms = vocabulary - set(self.embedding.index)
        n = len(new_terms)
        if n > 0:
            dim = self._dimensionality()

            # generate matrix of tiny random embedding weights (objs x dims)
            # needed to avoid vanishing gradients in further trainings.
            # ensure each of the new obj vector is unique via incremental index
            # on the 1st dimension.
            df = pd.DataFrame(
                np.c_[
                    np.arange(0, 1, step=1 / n) / 10,      # incremental index.
                    (np.random.rand(n, dim - 1) - 1) / 10  # rnd rest between ~(-0.1, +0.1)
                ],
                dtype=self.resolution,
                index=new_terms
            )
            self.embedding = pd.concat([self.embedding, df], axis=0)

        return self

    # Shrinks rows & columns of an embedding as a way to reduce the memory
    # footprint & noise. Note however that this transformation is irreversible
    # & may affect embedding quality.
    # @return vocabulary: optional subset of objects to keep.
    # @return embed_dim:  optional fraction or number of dimensions to keep.
    # @return self
    def compact(self, vocabulary: set = None, embed_dim=None):
        if vocabulary:
            unk_vtr_name = self._unknown_vtr().name
            require(
                unk_vtr_name not in vocabulary,
                f"cannot remove the unknown vector: '{unk_vtr_name}'"
            )

            # Identify terms to remove (include unknown vector)
            removable_terms = set(self.embedding.index) - vocabulary

            # update unknown vector with average of removable terms.
            # Note, this is biased & lossy on repeats
            self.embedding.loc[unk_vtr_name] = self.embedding[
                self.embedding.index.isin(removable_terms)
            ].mean()

            # Remove unwanted vocabulary but keep unknown vector.
            self.embedding = self.embedding.drop(
                removable_terms - {unk_vtr_name},
                errors='ignore'
            )

        if embed_dim:
            require(embed_dim > 0, "positive fraction or number of dimensions needed")
            self.embedding = self._update_embedding(
                PCA(n_components=embed_dim).fit_transform(self.embedding)
            )

        return self

    # Serializes embedding as a compressed TSV archive.
    def save(self, filename: str):
        self.embedding.to_csv(
            filename,
            sep='\t',
            compression=self.compression_format,
            index=True,
            header=False,
        )

        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

        return self

    # USAGE METHODS #

    # Looks up an embedding vector for the object. Out-of-vocabulary objects
    # produce a unique & deterministic random permutation of the UNKNOWN vector.
    # @param obj:   object key
    # @return vtr
    @cached(cache=LRUCache(maxsize=128))
    def vectorize(self, obj: str = None) -> pd.Series:
        require(not self.embedding.empty, "this embedding needs learning")

        if self.has_vocab(obj):
            vtr = pd.Series.copy(self.embedding.loc[obj], deep=True)
        elif obj is None:
            vtr = pd.Series.copy(self._unknown_vtr())
        else:
            unk_id = int(hashlib.md5(str(obj).encode('utf-8')).hexdigest(), 16)
            col_idx = unk_id % self._dimensionality()

            random.seed(unk_id)
            vtr = pd.Series.copy(self._unknown_vtr(), deep=True).rename(obj)
            unique = False
            while not unique:
                vtr.at[col_idx] = random.uniform(-1, 1)
                unique = not self._exists_mask(vtr).any()
        return vtr

    # Inverse of vectorize to find the closest object from a possibly non
    # existing vector.
    # @param vtr:   vector to identify, its name might be bogus.
    # @return obj
    @cached(cache=LRUCache(maxsize=64), key=custom_keyable_args)
    def objectify(self, vtr: pd.Series) -> str:
        require(not self.embedding.empty, "must first load an embedding")

        hits = self.embedding[self._exists_mask(vtr)]
        if hits.shape[0] == 1:
            result = hits.index
        else:
            result = self.rank(vtr, n=1)
        return result[0]

    # Finds top most similar vectors listed in descending relevance order.
    # @param objvtr: object key or vector.
    # @param n:      number of results to retrieve.
    # @return list of nearest objects.
    @cached(cache=LRUCache(maxsize=64), key=custom_keyable_args)
    def rank(self, objvtr, n: int = 10) -> list:
        require(n > 0, "need number of results to fetch")

        if isinstance(objvtr, pd.Series):
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
            n_neighbors=n + 1,
            return_distance=False,
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
        )[:n]

    # Compares cosine similarity between vectors.
    # @param objvtr1:  one object key or vector.
    # @param objvtr2:  another object key or vector.
    # @return sim:     1 = synonyms, 0 = irrelevant, -1 = antonyms
    @cached(cache=LRUCache(maxsize=64), key=custom_keyable_args)
    def similarity(self, objvtr1, objvtr2) -> float:
        objvtr1_strtype = isinstance(objvtr1, str)
        objvtr2_strtype = isinstance(objvtr2, str)

        if objvtr1_strtype and objvtr2_strtype and objvtr1 == objvtr2:
            result = 1.0
        else:
            vtr1 = (self.vectorize(objvtr1) if objvtr1_strtype else objvtr1).to_numpy()
            vtr2 = (self.vectorize(objvtr2) if objvtr2_strtype else objvtr2).to_numpy()

            numerator = np.dot(vtr1, vtr2)
            denominator = np.linalg.norm(vtr1) * np.linalg.norm(vtr2)
            result = numerator / denominator if denominator != 0 else 0
        return result

    # Public helper to check if object is known to the embedding.
    def has_vocab(self, obj: str) -> bool:
        return obj in self.embedding.index

    # PRIVATE METHODS #

    @staticmethod
    def _build_sgns(vocab_size: int, embed_dim: int, weights: np.ndarray = None):
        require(embed_dim < vocab_size, "dimension cannot be larger than vocabulary")

        shape = (1,)
        input_target = Input(shape, name='target_obj')
        input_context = Input(shape, name='context_obj')

        def build_embedding(input_data: Input, name: str):
            lyr = Embedding(
                input_dim=vocab_size,
                output_dim=embed_dim,
                input_length=1,
                weights=None if weights is None else [weights],
                name=f'{name}_embed',
            )(input_data)
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

    # Splits skip-grams into array of its tuple parts. Reverse maps objects
    # to embedding's vocabulary indices filtering any unknown gram.
    def _parse_skipgrams(self, skip_grams: list) -> Tuple[list, list, list]:
        targets, contexts, labels = zip(*skip_grams)
        targets = list(targets)
        contexts = list(contexts)
        for i in range(len(skip_grams)):
            targets[i] = self._index_of(targets[i])
            contexts[i] = self._index_of(contexts[i])
        return targets, contexts, labels

    # Dataframe mask matching to row of values.
    def _exists_mask(self, vtr: pd.Series):
        require(
            vtr.shape == (self._dimensionality(),),
            f"vector {vtr.shape} needs same dimensionality as embedding {self._dimensionality()}"
        )
        keep = 1
        for col in self.embedding.columns:
            keep = keep & (self.embedding[col] == vtr[col])
        return keep

    # Latent concept size of the embedding.
    def _dimensionality(self) -> int:
        return self.embedding.shape[1] if not self.embedding.empty else 0

    # Index of the obj if it exists, else 0 as the unknown vector.
    def _index_of(self, obj: str) -> int:
        return \
            0 if self.has_vocab(obj) else \
            self.embedding.index.get_loc(obj)

    def _update_embedding(self, data):
        df_embed = pd.DataFrame(data)  # idx->vtr
        df_terms = self.embedding \
            .drop(self.embedding.columns, axis=1).reset_index() \
            .rename(columns={"index": self.index_col})  # idx->obj
        return df_terms.merge(
            df_embed,
            left_index=True,
            right_index=True,
        ).set_index(self.index_col)  # obj->vtr

    # Symbolic vector for the out-of-vocabulary
    def _unknown_vtr(self) -> pd.Series:
        return self.embedding.iloc[0]   # assumed!

    # MEMBER VARIABLES #

    index_col = 'object'
    compression_format = 'xz'
    checkpoint_file = '.obj2vec.h5'
    resolution = np.float64


# UTILITIES #
    
# Assert like helper to validate inputs at runtime.
def require(condition: bool, msg: str = ""):
    if not condition:
        raise ValueError(msg)
    return

def custom_keyable_args(*args, **kwargs) -> tuple:
    key = tuple()

    for arg in args:
        try:
            key += as_immutable(arg)
        except TypeError:
            key += hashkey(arg)

    try:
        key += as_immutable(kwargs)
    except TypeError:
        key += hashkey(kwargs)

    return key

# Recursively converts an object into an immutable tuple representation
def as_immutable(obj) -> tuple:
    # TODO python object?

    if isinstance(obj, dict):
        return tuple(
            (k, as_immutable(v)) for k, v in sorted(obj.items())
        )

    if isinstance(obj, set):
        return tuple(
            as_immutable(v) for v in sorted(obj)
        )

    if isinstance(obj, list):
        return tuple(
            as_immutable(v) for v in obj
        )

    if isinstance(obj, pd.Series):
        return (
            obj.name, as_immutable(obj.to_list())
        )

    return obj,
