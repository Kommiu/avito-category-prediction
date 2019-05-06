### Стуктура проекта:

1. Вспомогательные файлы
    * [models.py](https://github.com/Kommiu/avito-category-prediction/blob/master/models.py)
    pytorch модели: lstm и cnn для обработки текста с опциональным 
    добавлением численных фичей
    * [params.py](https://github.com/Kommiu/avito-category-prediction/blob/master/params.py)
    константы используемые всеми тетрадками
    * [training_utils.py](https://github.com/Kommiu/avito-category-prediction/blob/master/training_utils.py)
    функции для обучения и кроссвалидации pytorch моделей, 
    а также различные вспомогательные функции
2. Тетрадки
    * [notebooks/label_hierarchies] генерация отображения (почему то оказалсь заигнорена git'ом)
    __category_id__ лейблов в лейблы других уровней иерархии
    *  [notebooks/process_data](https://github.com/Kommiu/avito-category-prediction/blob/master/notebooks/process_data.ipynb) обработка исходных данных и запись 
    в удобные для обучения форматы
    * [notebooks/fastText](https://github.com/Kommiu/avito-category-prediction/blob/master/notebooks/fastText.ipynb)  эксперименты с моделями на основе fastText
    *  [notebooks/tfidf](https://github.com/Kommiu/avito-category-prediction/blob/master/notebooks/tfidf.ipynb) эксперименты с tf-idf фичами и простыми
     моделями из sklearn
    * [notebooks/rnn_cross_val](https://github.com/Kommiu/avito-category-prediction/blob/master/notebooks/rnn_cross_val.ipynb) эксперименты с lstm моделями
    * [notebooks/cross_val_сnn](https://github.com/Kommiu/avito-category-prediction/blob/master/notebooks/cross_val_cnn.ipynb) неактуальный код для экспериментов 
    с cnn моделями, не вышло получить модель хотябы сравнимую с lstm, поэтому я не стал приводить тетрадку в актуальный вид
    * [notebooks/validation](https://github.com/Kommiu/avito-category-prediction/blob/master/notebooks/validation.ipynb)  проверка финальных моделей 
    на отложенной выборке
    * [notebooks/make_test_predictions](https://github.com/Kommiu/avito-category-prediction/blob/master/notebooks/make_test_predictions.ipynb) обучаю lstm на 
    всей тренировочной выборке и делаю предсказания, но неуспеваю.
    * [notebooks/final_preds](https://github.com/Kommiu/avito-category-prediction/blob/master/notebooks/final_preds.ipynb) экстренное предсказание для теста с помощью tf-idf + LinearSVC
3. Предсказания для тестовой выборки:
[predictions.csv](hhtps:)

Я использовал также pretrained fastText embeddings от [deeppavlov.ai]:
```console
whet -O embeddings/ft_native_300_ru_wiki_lenta_lemmatize.vec http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_lemmatize/ft_native_300_ru_wiki_lenta_lemmatize.vec
```


### Результаты
Accuracy на отложенной выборке (файлы data/**/valid*) 
в порядке увеличения вложенности (последний уровень -- данные __category_id__)
* Для lstm модели:
1. 0.9636787056708612
2. 0.9450686386664487
3. 0.8911586860598137
4. 0.8873079751593398
* для tf-idf+LinearSVM:
1. 0.9655478836411179
2. 0.9490419186141527
3. 0.8969807158032358
4. 0.8933751429972218

Результаты для LinearSVC получились лучше, но в финальной посылке я все равно хотел засылать предсказания lstm модели.
Но кто-то знаял всю память на гпу сервера и я не успел предсказать, поэтому таки отсылаяю SVC 
Изначально я хотел сделать простой ансамбль с мажоритарным голосванием.
Но не хватило времени отобрать подходящие модели. Хотел выбрать модели 
у  которых метрика  Жакарада по множествам неправильно 
предсказаных примеров была бы минимальна.
