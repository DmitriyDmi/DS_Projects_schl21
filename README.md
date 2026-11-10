# DS Проекты

### Описание проектов

Проекты, выполненные в рамках обучения в Школе 21

|Проект|Описание|Стек|Метрика|NoteBooks|
|-|-|-|-|-|
|DS1\_Tweets|Классификация твитов по тональности (3 класса) и поиск похожих текстов: токенизация, стемминг, лемматизация, исправление опечаток; BoW, word counts, TF-IDF, word2vec; косинусная близость.|pandas, nltk, pyspellchecker, scikit-learn, gensim, scipy|accuracy 0.90 (GridSearchCV)|[DS1\_Tweets](https://github.com/DmitriyDmi/DS_Projects_schl21/blob/main/DS1_Tweets/src/main.ipynb)|
|DS2\_Churn\_prediction|Прогноз оттока банковских клиентов (355 тыс. записей, дисбаланс 8%): предобработка, импутация, IsolationForest, сравнение 6 моделей — от бейзлайна до MLP на NumPy с ручным backprop.|pandas, scikit-learn, TensorFlow/Keras, NumPy|ROC AUC 0.83 (Keras)|[DS2\_Churn\_prediction](https://github.com/DmitriyDmi/DS_Projects_schl21/blob/main/DS2_Churn_prediction/src/main.ipynb)|
|DS3\_My\_Spotify|Рекомендации музыки на Million Song Dataset (1 млн прослушиваний): top-N по популярности и жанрам, контентная фильтрация по лирике (word2vec), коллаборативная фильтрация на implicit ALS|pandas, numpy, scipy, implicit, gensim|Precision@10: 0.12 (full) / 0.22 (когорта)|[DS3\_My\_Spotify](https://github.com/DmitriyDmi/DS_Projects_schl21/blob/main/DS3_My_Spotify/src/main.ipynb)|
|DS4\ Understanding\_customer|Классификация интентов чат-бота (7 классов): сравнение SimpleRNN, LSTM, CNN и fine-tuned BERT; fallback-механика с подбором порога уверенности|PyTorch, transformers (BERT), Keras, sklearn|accuracy 0.95 / F1 0.95 (BERT)|[DS4\_Understanding\_customer](https://github.com/DmitriyDmi/DS_Projects_schl21/blob/main/DS4_Understanding_customer/src/main.ipynb)|
|DS5\_Uber|Прогноз спроса на такси по 78 зонам (15-мин интервалы, горизонт 7 дней): наивные бенчмарки, скользящее среднее, Holt-Winters, SARIMA, LightGBM, MLP | pandas, numpy, statsmodels, LightGBM, TensorFlow/Keras | MAE 0.56 |[DS5\_Uber](https://github.com/DmitriyDmi/DS_Projects_schl21/blob/main/DS5_Uber/src/main.ipynb)|
|DS6\_City\_life|Геопространственный анализ: кластеризация зон, построение карт | geopandas, folium | кластеризация и карты |[DS6\_City\_life](https://nbviewer.org/github/DmitriyDmi/DS_Projects_schl21/blob/main/DS6_City_life/src/main.ipynb)|
<details>
|DS7\_Amazon|Введение в Apache Spark (Python API) и анализ сетей и графов.|||[DS7\_Amazon]()|
|DS8\_Fried\_Eggs|Введение в глубокое обучение и компьютерное зрение: CNN, сегментация объектов, детекция объектов, разметка данных.|||[DS8\_Fried\_Eggs]()|
</details>
