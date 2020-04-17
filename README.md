![Python 3.7](maupassant/website/static/img/logo/logo.png)

![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)
![Python package](https://github.com/Jwuthri/Maupassant/workflows/Python%20package/badge.svg)
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
import os

import pandas as pd

from maupassant.classifier.train import Trainer
from maupassant.settings import DATASET_PATH


train_path = os.path.join(DATASET_PATH, "sentiment_train.csv")
test_path = os.path.join(DATASET_PATH, "sentiment_test.csv")
val_path = os.path.join(DATASET_PATH, "sentiment_val.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
val_df = pd.read_csv(val_path)

# To train binary model which predict only 1 classe over 2, here the example predict positive/negative
train = Trainer(train_df, test_df, val_df, "binary-label", "CNN_NN", "feature", "binary", epochs=5, multi_label=False)
model_path = train.main()
# results = ["Ok": "positive", "I don't like this": "negative", "I like it": "positive", "Fuck you": "negative"]

# To train model which can predict 1 classe over (n), here the example predict insult/negative/neutral/obscene/offensive/positive/toxic
train = Trainer(train_df, test_df, val_df, "single-label", "CNN_NN", "feature", "single", epochs=5, multi_label=False)
model_path = train.main()
# results = ["Ok": "neutral", "I don't like this": "negative", "I like it": "positive", "Fuck you": "insult"]

# To train multi-label model which can predict (n) classes over (n), here the example insult/negative/neutral/obscene/offensive/positive/toxic
train = Trainer(train_df, test_df, val_df, "multi-label", "CNN_GRU_NN", "feature", "multi", epochs=5, multi_label=True)
model_path = train.main()
# results = ["Ok": "neutral", "I don't like this": "negative", "I like it": "positive", "Fuck you": ("negative", "toxic", "insult")]

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

