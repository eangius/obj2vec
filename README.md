# Object-2-Vector
General purpose utility to machine-learn distributed semantic vector spaces from arbitrary objects.
Key features include:

* Decouples domain contexts from embedding generation & query mechanics.
* Can build embeddings from scratch or customize from public ones.
* Robust handling out-of-vocabulary objects at predict time.
* Efficiently find synonyms, antonyms or irrelevant objects.
* Allows for vector arithmetic to find associations between objects.

------------------
### Contextualizing Objects
The key to building good embeddings is to identify relevant objects & extract
meaningful context from the data set which varies from domain to domain. To use
this library callers must generate an object `vocabulary` as well as `skipgram`
contexts explained below.

#### Object Vocabulary
An object to this library is anything that can satisfy the following properties:
```text
[1] Objects are identifiable via a unique string.
    — Caller can serialize or assign own ID scheme.
    — These IDs are how library interacts with the objects.

[2] Objects are of the same type & immutable within the domain.
    — Automatic relationship learning treats all objects equally.
    — Once an object is defined as a vector it is not expected to evolve.
      Callers may want to manage object versions externally.

[3] Objects have positional context with other objects.
    — ex: (textual objects) words/phrases in documents.
    — ex: (structured objects) products in customer purchase history.
    — ex: (spatial objects) point-of-interest on a map.
```

Before using this library, the caller needs to produce from their data set a
`vocabulary` list with the universe of unique objects to vectorize in the
embedding. The universe needs not to be complete & some objects may be filtered
out. This is why index 0 is reserved to let you encode the _catch all_ unknown
object. For example, if embedding the classical word2vec model the vocabulary
would look as follows.

```python
my_corpus = ['the cat jumped over the moon']
my_vocabulary = ['<???>', 'the', 'cat', 'jumped', 'over', 'moon', ...]
```

#### Building Skip-Grams
To learn embeddings, the library needs data formatted into a list of skip grams
of the form `([(target, context)], labels)` where `labels` is an array of numerical
flag indicating whether each of the `target` objects belongs with the corresponding
`context` object or not.

How to extract `target`/`context` pairs really depends on your domain. This
essentially involves sampling & filtering both co-occurring & never occurring
objects from your data set. In many cases, the positive/negative `labels` can be
automatically derived in an unsupervised manner with a sliding/convolution window.

For example, embedding the classical word2vec model on `my_corpus` with a sliding
window of 1 would produce these skip grams to train the model.
```python
my_skipgrams = [(
    [['jumped', 'the'], ['jumped', 'cat'], ['over', 'jumped'], ..., ['the', '<???>'], ['moon', 'over'], ['the', 'over']],
    [0, 1, 0, ..., 0, 0, 1]
)]
```
------------------
### Using Embeddings
You can load existing embeddings into your application as follows: 
```python
from embeddings.obj2vec import *

model = Obj2Vec(filename=f'data/embedding_glove.6B.80d-filtered.tsv.zip')
model.objectify(
    model.vectorize('king') -
    model.vectorize('man') +
    model.vectorize('woman')
) == 'queen'
```
The `glov.6B.80d-filtered.tsv.zip` here is a pruned version of a public google
pre-trained embedding on a huge corpus of data. You can use [others](https://nlp.stanford.edu/projects/glove/)
or even load your own vectors too! The library lets you convert objects to/from
vectors & gives you handy ways to operate with them by reference. In this case
to find concept associations via arithmetic!

Take a look also at the `.rank()` & `.similarity()` methods for other ways to
explore these vector spaces. A common use-case is to `.vectorize()` objects &
feed them to downstream machine learning tasks.

------------------
### Building New Embeddings
To build embeddings with your own data from scratch you can run:
```python
from embeddings.obj2vec import *

model = Obj2Vec(vocabulary=my_objects)
model.build(embed_dim=10)               # hyper param
model.learn(my_skipgrams, epochs=5)     # hyper param
model.save('new_embedding.tsv.zip')
```
Generating new embeddings is useful when working in specialized domains & have
enough data to generate the `my_object` & `my_skipgrams` variables to learn from.
This is a one time machine learned task that is usually done offline. There are
several other hyper parameters in the `.build()` & `.learn()` methods to fine
tune resulting vectors to your specific data & computing resources.

------------------
### Customizing Existing Embeddings
You can also improve & customize existing embeddings like so:
```python
from embeddings.obj2vec import *

model = Obj2Vec(filename=f'data/embedding_glove.6B.300d.tsv.zip')
model.compact(vocabulary=my_objects)                      # optional
model.build(embed_dim=10)                                 # hyper param
model.learn(my_skipgrams, epochs=5)                       # hyper param
model.append(Obj2Vec(filename='other_embedding.tsv.zip')) # optional
model.save('custom_embedding.tsv.zip')
```
This transfer learning approach pays off when your domain is a variation of an
existing one or you have limited data to learn from scratch. The `.compact()`
method in particular is handy to reduce the vocabulary size or dimensionality
of the existing embedding to better fit your use-case. This helps save space,
speed up training & remove noise at search time.

You can also combine multiple embeddings into one via the `.append()` method.
Provided all embeddings are different & have the same set of objects, this
technique allows you to enrich objects with different forms of semantics.
