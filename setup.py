from setuptools import setup

setup(
    name='obj2vec',
    version='1.0',
    description='General purpose object to vector machine learned embedding building module.',
    keywords='ml embedding',
    url='https://github.com/eangius/obj2vec',
    license='MIT',
    author='Elian Angius',
    packages=['obj2vec'],
    data_files=[
        ('data/', ['data/embedding_glove.6B.80d-filtered.tsv.zip'])
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'keras',
        'pandas',
        'pytest',
        'tensorflow',
        'scikit-learn',
    ],
)
