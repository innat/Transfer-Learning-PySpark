
from utils import plot_confusion_matrix, multiclass_roc_auc_score
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
import numpy as np
import itertools

# create spark session
spark = SparkSession.builder.appName('BD Recognizer').getOrCreate()

# loaded image
zero_df = ImageSchema.readImages("images/0").withColumn("label", lit(0))
one_df = ImageSchema.readImages("images/1").withColumn("label", lit(1))
two_df = ImageSchema.readImages("images/2").withColumn("label", lit(2))
three_df = ImageSchema.readImages("images/3").withColumn("label", lit(3))
four_df = ImageSchema.readImages("images/4").withColumn("label", lit(4))
five_df = ImageSchema.readImages("images/5").withColumn("label", lit(5))
six_df = ImageSchema.readImages("images/6").withColumn("label", lit(6))
seven_df = ImageSchema.readImages("images/7").withColumn("label", lit(7))
eight_df = ImageSchema.readImages("images/8").withColumn("label", lit(8))
nine_df = ImageSchema.readImages("images/9").withColumn("label", lit(9))


# merge data frame
dataframes = [zero_df, one_df, two_df, three_df,
              four_df,five_df,six_df,seven_df,eight_df,nine_df]

df = reduce(lambda first, second: first.union(second), dataframes)

# repartition dataframe
df = df.repartition(200)

# split the data-frame
train, test = df.randomSplit([0.8, 0.2], 42)

print(df.toPandas().size)
print(df.printSchema())


'''
--------------------------------- Model Building & Training -----------------------------------
'''


# model: InceptionV3
# extracting feature from images
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features",
                                 modelName="InceptionV3")

# used as a multi class classifier
lr = LogisticRegression(maxIter=5, regParam=0.03,
                        elasticNetParam=0.5, labelCol="label")

# define a pipeline model
sparkdn = Pipeline(stages=[featurizer, lr])
spark_model = sparkdn.fit(train)


'''
--------------------------------- Model Evaluation -----------------------------------
'''

# Evaluation Matrix
evaluator = MulticlassClassificationEvaluator()
transform_test = spark_model.transform(test)

print('F1-Score ', evaluator.evaluate(transform_test,
                                      {evaluator.metricName: 'f1'}))
print('Precision ', evaluator.evaluate(transform_test,
                                       {evaluator.metricName: 'weightedPrecision'}))
print('Recall ', evaluator.evaluate(transform_test,
                                    {evaluator.metricName: 'weightedRecall'}))
print('Accuracy ', evaluator.evaluate(transform_test,
                                      {evaluator.metricName: 'accuracy'}))

#
# -----------------------------------------------------
# Confusion Matrix

'''
- Convert Spark-DataFrame to Pnadas-DataFrame
- Call Confusion Matrix With 'True' and 'Predicted' Label
'''

y_true = transform_test.select("label")
y_true = y_true.toPandas() # convert to pandas dataframe from spark dataframe

y_pred = transform_test.select("prediction")
y_pred = y_pred.toPandas() # convert to pandas dataframe from spark dataframe

cnf_matrix = confusion_matrix(y_true, y_pred,labels=range(10))

sns.set_style("darkgrid")
plt.figure(figsize=(7,7))
plt.grid(False)

# call pre defined function
plot_confusion_matrix(cnf_matrix, classes=range(10))

#
# -----------------------------------------------------
# Classification Report

'''
- Classification Report of each class group
'''
target_names = ["Class {}".format(i) for i in range(10)]
print(classification_report(y_true, y_pred, target_names = target_names))


#
# -----------------------------------------------------
# ROC AUC Score

'''
- A custom ROC AUC score function for multi-class classification problem
'''


print('ROC AUC score:', multiclass_roc_auc_score(y_true,y_pred))


#
# -----------------------------------------------------
# Sample Prediction

'''
- Comparing true vs predicted samples
'''
# all columns after transformations
print(transform_test.columns)

# see some predicted output
transform_test.select('image', "prediction", "label").show()
