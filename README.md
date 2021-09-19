# AIIJC

## Коротко о работе:
Всю проделанную нами работу можно разделить на несколько основных шагов:
1. Собрать данные
2. Очистить и нормализовать данные
3. Эксперименты, эксперименты, и ещё раз эээээкспериименты
4. Создание телеграм-бота

Расскажем поподробнее о каждом из этих шагов

## Сбор данных

Для обучения наших моделей мы решили собрать дополнительные данные, так как тексты из википедии не очень похожи на школьные задания.

1. **Сборники задач**
После изучения тестового датасета мы поняли, что нашими целевыми заданиями являются задачки по математике для детей начальной школы.
Это задачки из разряда "У белочки было 3 орешка, два она потяряла. Сколько орехов осталось у белочки?". Поэтому за основу создания датасета были взяты 
[сборники](https://github.com/sergak0/AIIJC/tree/main/Sergey/data_zadachi) задач по математике для 1-5 класса. Их мы взяли из открытых источников в интернете.

После того как данные были скачаны и [привидены в нормальный вид](https://github.com/sergak0/AIIJC/blob/main/Sergey/Additional_dataset_creating.ipynb) (достали сами условия задач), мы прогнали их байесовской модели, обученной к тому времени на википедии. Но, к сожалению, результаты обучения на таком датасете нас не порадовали и поэтому пришлось размечать обучающие данные ручками.

2. **Интересные факты**
Также мы подумали, что различные интересные факты, которые можно найти в интернете (например, по запросу "Итересные факты о животных"), довольно чистые и достаточно неплохо походят на задачи из тестового датасета, а также их не надо размечать. Мы ручками собрали их, заходя на различные сайты.

## Предобработка данных

- Токенизация
- Выброс чисел
- Лемматизация с помощью pymorphy2
- Очистка от стоп-слов (не несущих смысла, вроде местоимений и предлогов)
- Выделение биграмм


## Исследование

В ходе решения кейса мы перепробовали несколько подходов и архитектур, расскажем поподробнее о каждом и них.
1. **Классические алгоритмы машинного обучения:**
- Наивный байес: был лидерм по точности, пока обучали только на википедии. 
Объясняется, вероятно, тем, что из-за отсутствия какой либо структуры в тексте остальные модели плохо с ним работали. 
А Байес, как статистическая модель, выезжал за счёт разнообразия лексики с статьях википедии.
Но после дообучения всех моделей на задачах и фактах стал отставать от своих собратьев, так как лексику не сильно пополнил, а структуру так и не усвоил
- Логистическая регрессия на one-hot encoded dictionary: плохое качество, тк вектора очень разреженные
- [Логистическая регрессия на tf-idf ecncoded](https://github.com/sergak0/AIIJC/blob/main/Sergey/logistic_regression.py): качество тоже так себе. [Usage](https://github.com/sergak0/AIIJC/blob/main/Sergey/Logistic_regression_usage.ipynb)
- Логистическая регрессия на Word2Vec: лишь немного уступал Байесу на вики-данных, на чистых данных перегнал его и является неплохим бейзлайном, 
тк учитывает общий смысловой вектор текста
- Логистическая регрессия/полносвязная нейронная сеть на LDA: в числе лидеров на вики-данных, потрясающе самостоятельно выделяет нужные нам 4 класса и пятый со стоп-словами, не несущими смысла. 
Использовали для создания хорошего словаря стоп-слов
- Всё то же самое, но смотрим только на ключевые слова, выделенные на основе cosine similarity векторов: точность модели почти не меняется, таким образом, мы можем "сжать" данные и заставить конечный алгоритм работать быстрее.

2. **Трансформеры**
- BERT: мы использовали bert-base-multilingual-uncased. На википедии очень плохо обучался, потому что данные из википедии очень грязные. На дополнительных данных, которые мы собирали, показал себя лучше всех моделей. Пробовали обучать на сборниках задач, на фактах, на сборниках задач и фактах вместе. Последнее дало лучший результат. С помощью output_attentions=True мы выводили ключевые слова.
 
 3. **Подходы и открытия**
 - В качестве входный данных использовать не текст, а ключвые слова, выделенные на основе cosine similarity векторов слов: 
 точность модели почти не меняется, таким образом, мы можем "сжать" данные и заставить конечный алгоритм работать быстрее.
