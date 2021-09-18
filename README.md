# AIIJC

## Коротко о работе:
Всю проделанную нами работу можно разделить на несколько основных шагов:
1. Собрать данные
2. Очистить и нормализовать данные
3. Эксперименты, эксперименты, и ещё раз эээээкспериименты
4. Создание телеграм-бота

Расскажем поподробнее о каждом из этих шагов

## Сбор данных

-- TO DO --

## Предобработка данных

-- TO DO --

## Исследование

В ходе решения кейса мы перепробовали несколько подходов и архитектур, расскажем поподробнее о каждом и них.
1. **Классические алгоритмы машинного обучения:**
- Наивный байес: был лидерм по точности, пока обучали только на википедии. 
Объясняется, вероятно, тем, что из-за отсутствия какой либо структуре в тексте остальные модели плохо с ним работали. 
А Байес, как статистическая модель, выезжал за счёт разнообразия лексики с статьях википедии.
Но после дообучения всех моделей на задачах и фактах стал отставать от своих собратьев, так как лексику не сильно пополнил, а структуру так и не усвоил
- Логистическая регрессия на one-hot encoded dictionary: плохое качество, тк вектора очень разреженные
- Логистическая регрессия на Word2Vec: лишь немного уступал Байесу на вики-данных, на чистых данных перегнал его и является неплохим бейзлайном, 
тк учитывает общий смысловой вектор текста
- Логистическая регрессия/полносвязная нейронная сеть на LDA: в числе лидеров на вики-данных, потрясающе самостоятельно выделяет нужные нам 4 класса и пятый со стоп-словами, не несущими смысла. 
Использовали для создания хорошего словаря стоп-слов
- Всё то же самое, но смотрим только на ключевые слова, выделенные на основе cosine similarity векторов: точность модели почти не меняется, таким образом, мы можем "сжать" данные и заставить конечный алгоритм работать быстрее.

2. **Трансформеры**
-BERT
 -- TO DO --
 
 3. **Подходы и открытия**
 - В качестве входный данных использовать не текст, а ключвые слова, выделенные на основе cosine similarity векторов слов: 
 точность модели почти не меняется, таким образом, мы можем "сжать" данные и заставить конечный алгоритм работать быстрее.
