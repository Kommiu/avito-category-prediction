### Стуктура проекта:

1.  Utility files
    * [models.py](https://github.com/Kommiu/avito-category-prediction/blob/master/models.py)
definition of pytorch models: lstm и cnn.
    * [params.py](https://github.com/Kommiu/avito-category-prediction/blob/master/params.py)
    constants used by all notebooks
    * [training_utils.py](https://github.com/Kommiu/avito-category-prediction/blob/master/training_utils.py)
    utility functions to facilitate sklearn-like training and cross-validation
2. Notebooks
    * [notebooks/label_hierarchies] analysis of category hierarchies .
    *  [notebooks/process_data](https://github.com/Kommiu/avito-category-prediction/blob/master/notebooks/process_data.ipynb)  data preprocessing
    * [notebooks/fastText](https://github.com/Kommiu/avito-category-prediction/blob/master/notebooks/fastText.ipynb)  fastText based models
    *  [notebooks/tfidf](https://github.com/Kommiu/avito-category-prediction/blob/master/notebooks/tfidf.ipynb) linear model with td-idf features
    * [notebooks/rnn_cross_val](https://github.com/Kommiu/avito-category-prediction/blob/master/notebooks/rnn_cross_val.ipynb) lstm-based models
    * [notebooks/cross_val_сnn](https://github.com/Kommiu/avito-category-prediction/blob/master/notebooks/cross_val_cnn.ipynb) cnn-based models
    * [notebooks/validation](https://github.com/Kommiu/avito-category-prediction/blob/master/notebooks/validation.ipynb)  validation of selected models

Pretrained fastText embeddings from [deeppavlov.ai]:
```console
whet -O embeddings/ft_native_300_ru_wiki_lenta_lemmatize.vec http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_lemmatize/ft_native_300_ru_wiki_lenta_lemmatize.vec
```


### Results
Accuracy on validation dataset on diferent hierarchy levels:
* lstm:
1. 0.9636787056708612
2. 0.9450686386664487
3. 0.8911586860598137
4. 0.8873079751593398
* tf-idf+LinearSVM:
1. 0.9655478836411179
2. 0.9490419186141527
3. 0.8969807158032358
4. 0.8933751429972218

