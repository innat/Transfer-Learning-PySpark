## Transfer Learning With PySpark

A promising solution for a **Computer Vision** problem with the power to combine state-of-the-art technologies: **Deep Learning** with **Apache Spark**. We will leverage the power of **Deep Learning Pipelines** for a Multi-Class image classification problem.

**Deep Learning Pipelines** is a high-level Deep Learning framework that facilitates common Deep Learning workflows via the **Spark MLlib** Pipelines API. It currently supports TensorFlow and Keras with the TensorFlow-backend. The library comes from Databricks.

**Blog Articles:** Read technical description from below sources.

- [Towards Data Science | Transfer Learning In PySpark](https://towardsdatascience.com/transfer-learning-with-pyspark-729d49604d45)

- [LinedIn-Article | Transfer Learning In PySpark](https://www.linkedin.com/pulse/transfer-learning-pyspark-mohammed-innat/)

---

### Installation

Installation is bit pesky right now. So, I decided to write it separately. Check this [GitHub-Gist](https://gist.github.com/iphton/b0ab252c954eb2a28a984774e3ee1f2d) to install necessary packages and technical stuffs.

### Data Set

We choose [**NumtaDB**](https://arxiv.org/abs/1806.02452) as a source of our datasets. It's a collection of Bengali Handwritten Digit data. The dataset contains more than **85,000** digits from over **2,700** contributors. But here we're not planning to work on the whole data set rather than choose randomly 50 images of each class. You can get the sample of some images from [here.](https://drive.google.com/open?id=1AbTGJIfD2lhGe-stNIymGaowy7vyVovn) It Contains 500 in total.

## Model Training

Here we combine the **InceptionV3** model and **logistic regression** in Spark. The **DeepImageFeaturizer** automatically peels off the last layer of a pre-trained neural network and uses the output from all the previous layers as features for the logistic regression algorithm.

## Evaluation

1. Evaluation matrix
```
F1-Score   0.81117
Precision  0.84220
Recall     0.80909
Accuracy   0.80909
```

2. Confusion Metrix
![Screenshot from 2019-07-23 00-40-15](https://user-images.githubusercontent.com/17668390/61664640-00afd880-acf5-11e9-8544-91b3e05fbbf4.png)


3. Classification Report
```
precision  recall   f1-score   support

     Class 0       1.00      0.92      0.96        13
     Class 1       0.57      1.00      0.73         8
     Class 2       0.64      1.00      0.78         7
     Class 3       0.88      0.70      0.78        10
     Class 4       0.90      1.00      0.95         9
     Class 5       0.67      0.83      0.74        12
     Class 6       0.83      0.62      0.71         8
     Class 7       1.00      0.80      0.89        10
     Class 8       1.00      0.80      0.89        20
     Class 9       0.70      0.54      0.61        13

   micro avg       0.81      0.81      0.81       110
   macro avg       0.82      0.82      0.80       110
weighted avg       0.84      0.81      0.81       110
```




**Predicted Samples**

```
['image', 'label', 'features', 'rawPrediction', 'probability', 'prediction']
+--------------------+----------+-----+
|               image|prediction|label|
+--------------------+----------+-----+
|[file:/home/i...|       1.0|    1|
|[file:/home/i...|       8.0|    8|
|[file:/home/i...|       9.0|    9|
|[file:/home/i...|       1.0|    8|
|[file:/home/i...|       1.0|    1|
|[file:/home/i...|       1.0|    9|
|[file:/home/i...|       0.0|    0|
|[file:/home/i...|       2.0|    9|
|[file:/home/i...|       8.0|    8|
|[file:/home/i...|       9.0|    9|
|[file:/home/i...|       0.0|    0|
|[file:/home/i...|       4.0|    0|
|[file:/home/i...|       5.0|    9|
|[file:/home/i...|       1.0|    1|
|[file:/home/i...|       9.0|    9|
|[file:/home/i...|       9.0|    9|
|[file:/home/i...|       1.0|    1|
|[file:/home/i...|       1.0|    1|
|[file:/home/i...|       9.0|    9|
|[file:/home/i...|       3.0|    6|
+--------------------+----------+-----+
only showing top 20 rows
```
