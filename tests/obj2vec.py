# Internal libraries
from obj2vec.embeddings import *

# Third party libraries
import pytest


model = Obj2Vec(filename=f'data/embedding_glove.6B.80d-filtered.tsv.zip')


# Unrecognized objects should have unique vectors similar to the unknown object.
def test__unknowns01():
    vtr1 = model.vectorize('buggywoman')  # not in vocab
    vtr2 = model.vectorize('akwa_man')    # not in vocab
    unk0 = model.vectorize('<???>')       # in vocab

    assert not vtr1.equals(vtr2)
    assert not vtr1.equals(unk0) and model.similarity(vtr1.name, unk0.name) > 0.9
    assert not vtr2.equals(unk0) and model.similarity(vtr2.name, unk0.name) > 0.9


# Converting from/to vectors should handle known & unknown objects.
def test__conversions01():
    assert model.objectify(model.vectorize('earth')) == 'earth'
    assert model.objectify(model.vectorize('_unknown_')) != '_unknown_'


# Related objects should have closer relative similarities than unrelated ones.
def test__similarity01():
    assert model.similarity('toronto', 'canada') > model.similarity('toronto', 'mexico')
    assert model.similarity('hydrogen', 'helium') > model.similarity('hydrogen', 'religion')


# Doing vector arithmetic should produce relevant vectors in the embedding.
def test__associations01():
    assert model.objectify(
        model.vectorize('king') -
        model.vectorize('man') +
        model.vectorize('woman')
    ) == 'queen'


# Getting top n results should retrieve relevant objects in decreasing order.
def test__rank01():
    assert model.rank('car') == ['driver', 'vehicle', 'driving', 'truck', 'drive', 'driven', 'auto', 'drove', 'automobile', 'taxi']
    assert model.rank('ice', n=3) == ['frozen', 'snow', 'melting']
    assert model.rank("_unknown_", n=1) == ['<???>']


# Compacting an embedding by vocabulary should only keep the common rows.
def test__compact01():
    model = Obj2Vec(filename=f'data/embedding_glove.6B.80d-filtered.tsv.zip')
    prev_rows, prev_cols = model.embedding.shape

    model.compact(vocabulary=['this', 'and', 'that', '_unknown_'])
    post_rows, post_cols = model.embedding.shape
    assert post_rows == 3 and post_cols == prev_cols


# Compacting an embedding by dimensions should only keep the main columns.
def test__compact02():
    model = Obj2Vec(filename=f'data/embedding_glove.6B.80d-filtered.tsv.zip')
    prev_rows, prev_cols = model.embedding.shape

    model.compact(dim_variance=8)
    post_rows, post_cols = model.embedding.shape
    assert post_rows == prev_rows and post_cols == 8
