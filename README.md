![Python 3.7](maupassant/website/static/img/logo/logo.png)

![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)
![Python package](https://github.com/Jwuthri/Maupassant/workflows/Python%20package/badge.svg)
![Package Status](https://img.shields.io/pypi/status/maupassant.svg)
-----------------

Maupassant is a suite of open source Python module for Natural Language. All the modules are really easy to use, but at the same time you can tune them easily. 

It can help you in differents ways:

* Preprocessing
    * Normalization
    * Tokenization

* Featuring
    * Clustering
    * Dimensional Reduction
    * Feature Extraction

* Modeling
    * Text Classification
    * Text Extraction
    * Text Similarity
    * Text Question Answer

##### Dependencies
```
- Python (>= 3.6)
- Tensorflow (>= 2.0.1)
- Nltk (>= 3.4.5)
- Scikit-learn (>= 0.22.2)
```

##### User installation
With pypi:
```pip install maupassant```

Custom installation:
```
git clone git@github.com:Jwuthri/Maupassant.git
cd maupassant
pip install -e .
```

Processing
----------
###### Normalization
```
* Word Stemming
* Word Lemming
* SpellCheck correction
* Remove Emoji
* Remove Emoticons
* Remove text contraction
```
###### Tokenization
```
* Sentence Tokenization
* Sentence DeTokenization
* Sequence Tokenization
* Sequence DeTokenization
```

Featuring
---------
###### Clustering
```
* HDBSCAN
* Kmeans
* AHC (Agglomerative-Hierarchical-Clustering)
* AE (Auto-Encoder)
* VAE (Variationnal-Auto-Encoder)
```
###### Dimensional Reduction
```
* LDA (Linear-Discriminant-Analysis)
* QDA (Quadratic-Discriminant-Analysis)
* PCA (Principal-Component-Analysis)
* SVD (Singular-Value-Decomposition)
* SPCA (Scaled-Principal-Component-Analysis)
* UMAP (Uniform-Manifold-Approximation-Projection)
* AE (Auto-Encoder)
* VAE (Variationnal-Auto-Encoder)
```
###### Feature Extraction
```
* TfIdf (Term-Frequency–Inverse-Document-Frequency)
* Embedding (MultiLanguage)
```

Modeling
---------
###### Text Classification
```
* 3 differents predefine models (work in multilanguage):
    * NN (basic)
    * CNN_NN (intermediate)
    * CNN_GRU_NN (advanced)
* Each models can work with:
    * 1 feature and predict 1 label
    * 1 feature and predict multiple labels
```

```python
from maupassant.classifier import task

task.train(train_path, test_path, val_path, feature, label, architecture, label_type)
```
###### Text Extraction
```
* Weighted TF-IDF
    * Here you need to provide a list of keywords (WIP)
* AutoEncoders
    * WIP
```
###### Text Similarity
```
* Cosine similarity between 2 documents
* For a given document find the most similar document from a list of documents
```
###### Text Question Answer
```
* For a given context, find the best answer for a question
```

