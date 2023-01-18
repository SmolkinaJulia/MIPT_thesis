# MIPT_thesis
# Магистерская диссертация
## Содержание репозитория
Thesis and science-related, research work during my masters at MIPT

В реопзитории сожержутся файлы-скрипты с экспериментами и некоторыми результатами, ответами на вопросы Гос эказамена, текст диссертации (будет добавлен позже) и другое.

## Основная информация
|ФИО обучающегося	            |Смолкина Юлия Александровна| -
| ------------- |:------------------:| -----:|
|Физтех-школа/факультет, группа|ФПМИ, ТИИ М06-106    | - 
|Базовая организация, кафедра	  |Научно-образовательный центр когнитивного моделирования| - 
|Тема НИР	                    |Разработка алгоритма классификации действий на видеоизображениях (Обучение эмбеддингов в задаче для нейросетевых моделей классификации) (Обучение эмбеддингов задач для нейросетевых моделей классификации)|Возможны изменения или дополнения 
|Наборы данных|mnist, fashoin mnist, pascal voc 2007, imagine net, cifar10|Дополняется 


## Вспомогательные материалы
> Референсы (статьи, теоретические материалы, исследования):
> 
> https://towardsdatascience.com/the-utility-of-task-embeddings-e00a18133f77
> https://github.com/cvxgrp/pymde/blob/main/examples/fashion_mnist.ipynb
> https://github.com/botkop/mnist-embedding/blob/master/notebooks/mnist-embedding-classifier.ipynb
> https://pymde.org
> https://github.com/tensorpack/tensorpack/blob/master/examples/SimilarityLearning/embedding_data.py
> https://keras.io/api/datasets/
> https://medium.com/coinmonks/how-to-get-images-from-imagenet-with-python-in-google-colaboratory-aeef5c1c45e5
> https://pymde.org/datasets/index.html
> https://github.com/MatthewWilletts/Embeddings/blob/master/make_embeddings.py
> https://habr.com/ru/post/666314/
> https://habr.com/ru/company/leader-id/blog/529012/
> https://vk.com/@papersreaders-graph-rise-graph-regularized-image-semantic-embedding
> https://arxiv.org/pdf/1902.10814.pdf

## Формальная и неформальная постановка задачи и теоретические выкладки
Что такое эмбеддинги и как они помогают искусственному интеллекту понять мир людей

Термин «эмбеддинг» (от англ. embedding – вложение) - стал часто встречаться в описаниях систем искусственного интеллекта только в последние несколько лет, а впервые появился в работах специалистов по обработке текстов на естественных языках. Естественный язык – это привычный способ общения людей. Например, язык машин – это двоичный код, в который компилируются все другие языки программирования. Однако в нашем случае речь идет именно об обработке естественного языка человека.

В русскоязычной литературе эмбеддингами обычно называют именно такие числовые векторы, которые получены из слов или других языковых сущностей. Напомню, что числовым вектором размерности k называют список из k чисел, в котором порядок чисел строго определен. Например, трехмерным вектором можно считать (2.3, 1.0, 7.35), а (1, 0, 0, 2, 0.1, 0, 0, 7.9) – восьмимерным числовым вектором.

Будем обучать модель для генерации эмбеддингов на задаче классификации таким образом, чтобы эмбеддинги похожих изображений (соседних вершин в графе) были как можно ближе друг к другу, то есть будем штрафовать модель если она ставит эмбеддинги похожих объектов далеко друг от друга.
Более формально, при обучении минимизируется следующая функция потерь (как пример):

где L(theta) — cross-entropy loss, Omega(theta) — регуляризатор, w_{u,v} — вес ребра, и d(.,.) — функция расстояния между эмбеддингами.

![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/theory/7zcHKeylpL0.jpg "alpha - hyperparameter")

Для того чтобы из предсказанного Image Embedding’a получить вероятности классов, эмбеддинг пропускается через полносвязный слой с последующим вычислением softmax’a.
Во время обучения вместе с размеченной картинкой u также выбирается картинка v, которая является соседней в графе, после чего вычисляется loss R(theta).
Важно отметить, что так как граф похожести используется только на этапе обучения, то время инференса остается прежним.


## Проведенные эксперименты и Текущее состояние НИР

Кратко опишу, какая работа была проделана, не вдаваясь в детали, подробнее можно посмотреть в разделе code. Там будет лежать код-скрипт, комментарии и визуализауия.

`1` pascal VOC 2007

- Проведена пред обработка данных

![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/practice/Снимок%20экрана%202023-01-18%20в%2017.22.39.png "Предобработка")

- Описано содержание через цвет, текстуру и форму (глобальные фичи)

![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/practice/Снимок%20экрана%202023-01-18%20в%2017.28.40.png "Описание содержимого")

- Детекция объкетов (локальные фичи)

![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/practice/Снимок%20экрана%202023-01-18%20в%2017.28.26.png "Детекция объкетов")

Для оценки качества использовалась модицикация IoU. Реалищовано только для крупнйешего класса - person.

`2` MNIST

- Проведене предобработка

![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/practice/Снимок%20экрана%202023-01-18%20в%2017.43.02.png "Предобработка")

- Кодирование меток-цифр с помощью one-hot encoding и получение векторов значений

![alt-текст]( "one-hot encoding")

- Тренировка и тест раззных классификаторов
- Как нейросетевая модель использоваласьс ерточная сеть с разными комбинациями слоёв и лосс функций
- Испольщование Sequence Graph Transform (SGT)

*SGT is a sequence embedding function. SGT extracts the short- and long-term sequence features and embeds them in a finite-dimensional feature space. The long and short term patterns embedded in SGT can be tuned without any increase in the computation.*

- Pymde Embeddings (Pymde - эмбеддинги реализованы с нуля)

*In these embeddings similar digits are near each other, and dissimilar digits are not near each other.*

![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/practice/Снимок%20экрана%202023-01-18%20в%2017.46.51.png "Pymde")

![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/practice/Снимок%20экрана%202023-01-18%20в%2017.46.58.png "Pymde")

![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/practice/Снимок%20экрана%202023-01-18%20в%2017.47.43.png "Pymde 3D")

- Валидация

![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/practice/Снимок%20экрана%202023-01-18%20в%2017.48.10.png "Валидация")

![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/practice/Снимок%20экрана%202023-01-18%20в%2018.00.38.png)


`3` Fashion MNIST

Аналогично Mnist по этому чразу приведу картинки результатов применения эмбеддинга

![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/practice/Снимок%20экрана%202023-01-18%20в%2017.55.47.png)

![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/practice/Снимок%20экрана%202023-01-18%20в%2017.55.19.png)
![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/practice/Снимок%20экрана%202023-01-18%20в%2017.55.29.png)
![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/practice/Снимок%20экрана%202023-01-18%20в%2017.55.36.png)
![alt-текст](https://github.com/SmolkinaJulia/MIPT_thesis/blob/main/pictures/practice/Снимок%20экрана%202023-01-18%20в%2017.55.39.png)


`4` Cifar10
*В процессе*

`5` Imagine Net
*В процессе*

### Используемые подходы и архитектуры

* FCNN
* Группа сетей VGG
* Группа сетей DenseNet
* Группа сетей ResNet
* Группа сетей ResNeXt
* Группа сетей ReXNet/ResNeSt/Res2Net
* Группа сетей RegNet
* Группа сетей Inception/Xception
* Группа сетей MNASNet/NASNet/PnasNet/SelecSLS/DLA/DPN
* Группа сетей MobileNet/MixNet/HardCoRe-NAS
* Группа сетей трансформеров BeiT/CaiT/DeiT/PiT/CoaT/LeViT/ConViT/Twins
* Группа сетей ViT (Visual Transofrmer)
* Группа сетей ConvNeXt
* Группа сетей ResMLP/MLP-Mixer
* Группа сетей NFNet-F
* Группа сетей EfficientNet
* Прочие предобученные модели сетей

### Метрики для оценивания качества результатов
Так как мы в том числе сравниваем работу нейросетевых подходов, это могут быть сверточные нейросети или что-то другое, то классические метрики как accuaracy и другие также применимы.

Немного более специфичными метриками для данной задачи будут:

`distance measures`

Мера расстояния обычно количественно определяет несходство двух векторов признаков. Мы вычисляем его как расстояние между двумя векторами в некотором метрическом пространстве.

* Manhattan distance,
* Mahalanobis distance, 
* Histogram Intersection Distance (HID)

`similarity metrics`

Метрика подобия количественно определяет сходство между двумя векторами признаков. Таким образом, это работает противоположно метрикам расстояния: наиболее значимое значение показывает изображение, похожее на изображение запроса.

Например, косинусное расстояние (cosine distance) измеряет угол между двумя векторами признаков.

`PA` Попиксельная точность (pixel accuracy)

`mPA` Средняя попиксельная точность по классам наблюдаемых объектов (mean pixel accuracy over classes)

`IoU` Метрика (Intersection over Union) или индекс Жаккара (Jaccard index) 

`Dice` Индекс Дайса (Dice index) или F1-score

А также набор других стандартных метрик

## План развития работы
`1` Добавление графовой архитектуры как входные данные

`2` Дополнение групп сетей ко всем наборам данных, которые еще не учавствовали в эксперименте

`3` Использование (добавление) более специфичных метрик (описанны выше)

`4` Проведение экспериментов на оставшихся датасетах (2шт)

`5` Создание единой сводной таблицы с результатами метрик по всем экспериментам - кол-ые эксперименты

`6` Качественные эксперименты: какие свойства, функционал есть у подходов и их возможные улучшения.

## Рекомендации по эксплуатации
`Дописать`
