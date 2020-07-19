```
     __  ___                                              __
    /  |/  /___ ___  ______  ____ _______________ _____  / /_
   / /|_/ / __ `/ / / / __ \/ __ `/ ___/ ___/ __ `/ __ \/ __/
  / /  / / /_/ / /_/ / /_/ / /_/ (__  |__  ) /_/ / / / / /_
 /_/  /_/\__,_/\__,_/ .___/\__,_/____/____/\__,_/_/ /_/\__/
                   /_/
```
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
```bash
- Python (>= 3.6)
- Tensorflow (>= 2.0.1)
- Nltk (>= 3.4.5)
- Scikit-learn (>= 0.22.2)
```

##### User installation
With pypi:
```pip install maupassant```

Custom installation:
```bash
git clone git@github.com:Jwuthri/Maupassant.git
cd maupassant
pip install -e .
```

Using docker:
```bash
docker-compose build
docker-compose run
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

```python
from maupassant.preprocessing.normalization import TextNormalization

tn = TextNormalization()

# text correction
text = 'Let me tell you somthing you alrady know.'
cleaned_text = tn.text_correction(text)
cleaned_text
>> 'Let me tell you something you already know.'

# Remove emoji/emot
text = 'Let me tell you something you already know 👍'
demojize_text = tn.text_demojis(text, how_replace="")
demojize_text
>> 'Let me tell you something you already know'

text = 'Let me tell you something you already know :)'
demoticons_text = tn.text_demoticons(text, how_replace="")
demoticons_text
>> 'Let me tell you something you already know '

# Decontract words
text = "I'd like to know yall guys"
decontraction_text = tn.text_decontraction(text)
decontraction_text
>> 'I would like to know you all guys'

# Stemming word
word = "shipping"
stemmed_word = TextNormalization().word_stemming(word)
stemmed_word
>> 'ship'

# example of sentence cleaning:
text = "I'd like to tell you somthing you alrady know."
decontraction_text = tn.text_decontraction(text)
cleaned_text = tn.text_correction(decontraction_text)
cleaned_text
>> 'I would like to tell you something you already know.'
```

###### Tokenization
```
* Sentence Tokenization
* Sentence DeTokenization
* Sequence Tokenization
* Sequence DeTokenization
```

```python
from maupassant.preprocessing.tokenization import SentenceTokenization, SequenceTokenization

# Sentence Tokenization
sentence = "Let me tell you something you already know."
tokens = SentenceTokenization().tokenize(sentence)
tokens
>> ['Let', 'me', 'tell', 'you', 'something', 'you', 'already', 'know', '.']

# Sentence DeTokenization
sentence = SentenceTokenization().detokenize(tokens)
sentence
>> 'Let me tell you something you already know.'

# Sequence Tokenization
sequence = "Let me tell you something you already know. The world ain’t all sunshine and rainbows."
sentences = SequenceTokenization().tokenize(sequence)
sentences
>> ['Let me tell you something you already know.', 'The world ain’t all sunshine and rainbows.']

# Sequence DeTokenization
sequence = SequenceTokenization().detokenize(sentences)
sequence
>> 'Let me tell you something you already know. The world ain’t all sunshine and rainbows.'
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
```
Be careful, the classifier works only with stacked column:

This way works:

| feature      | label    |
|--------------|----------|
| I love it    | positive |
| I don't know | neutral  |
| I hate you   | negative |

This way doesn't works:

| feature      | positive | negative | neutral |
|--------------|----------|----------|---------|
| I love it    | 1        | 0        | 0       |
| I don't know | 0        | 1        | 0       |
| I hate you   | 0        | 0        | 1       |

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
* 3 differents predefine models (work in multilanguage):
    * NN (basic)
    * GRU (intermediate)
    * CNN (intermediate)
    * CNN_LSTM (advanced)
    * CNN_GRU (advanced)
* Each models can work with:
    * 1 feature and predict 1 label, binary-label
    * 2 features and prediction 1 label, binary-label
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
