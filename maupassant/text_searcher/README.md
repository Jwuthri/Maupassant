Text searcher
=============

Train a model for question-answering, or documents similarity
(use paraphrase modeling)
```
txt1 => encoding1 => pooling1
txt2 => encoding2 => pooling2
concat(pool1, pool2, pool1 - pool2) => dense(sigmoid)
```