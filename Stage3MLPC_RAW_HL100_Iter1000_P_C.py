#Run code locally in Jupyter Notebook
#Stage3MLPC_RAW_HL100_Iter1000_P_C.py
import findspark
findspark.init()
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics
from prettytable import PrettyTable
from pyspark.sql.types import DoubleType
#import matplotlib.pyplot as plt
#import seaborn as sns

spark = SparkSession \
    .builder \
    .appName("Stage3MLPC_RAW_HL100_Iter1000_P_C") \
    .getOrCreate()

MLP_test_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Test-label-28x28.csv"
MLP_train_datafile = "hdfs://soit-hdp-pro-1.ucc.usyd.edu.au/share/MNIST/Train-label-28x28.csv"

MLP_train_labeldf = spark.read.csv(MLP_train_datafile,header=False,inferSchema="true")
MLP_test_labeldf = spark.read.csv(MLP_test_datafile,header=False,inferSchema="true")

assembler_train = VectorAssembler(inputCols = MLP_train_labeldf.columns[1:], outputCol = "features")
train_vectors_withlabel = assembler_train.transform(MLP_train_labeldf).selectExpr("_c0 as label", "features")

assembler_test = VectorAssembler(inputCols = MLP_test_labeldf.columns[1:], outputCol = "features")
test_vectors_withlabel = assembler_test.transform(MLP_test_labeldf).selectExpr("_c0 as label", "features")

train_vectors_withlabel

# specify layers for the neural network:
# input layer of size 28x28 = 784 (features), two intermediate of size 5 and 4
# and output of size 10 (classes)

#layers = [784, 5, 4, 10]

layers = [784, 100, 10]

trainer = MultilayerPerceptronClassifier(maxIter=1000, layers=layers, blockSize=128, seed=1234)
model = trainer.fit(train_vectors_withlabel)
result = model.transform(test_vectors_withlabel)
predictionAndLabels = result.select("prediction", "label")

changedTypedf = predictionAndLabels.withColumn("label", predictionAndLabels["label"].cast(DoubleType()))
test_rdd = changedTypedf.rdd.map(tuple)
metrics = MulticlassMetrics(test_rdd)




#Print F1-score, Recall and Precision for each label.

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

labels = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

x = PrettyTable(['Label', 'Precision', 'Recall', 'F1-score'])

for label in sorted(labels):
    x.add_row([label,round(metrics.precision(label),3),round(metrics.recall(label),3), round(metrics.fMeasure(label, beta=1.0),3)])
print(x)





#Print ConfusionMatrix for each label
#Below only works on Jupyter notebook, since we cannot see a seaborn confusion matrix
#directly from terminal



# sns.set() 

def display_cm(m):
     a = m.toArray().astype(np.float)
#     print(a)
#     print(m)
     row_sums = a.sum(axis=1)
     percentage_matrix = a.astype(np.float) / row_sums[:, np.newaxis]
     percentage_matrix =   100 *a.astype(np.float64) /a.astype(np.float64).sum(axis=1)
     print(percentage_matrix)
     
display_cm(metrics.confusionMatrix())
#     plt.figure(figsize=(10, 10))
#     sns.heatmap(percentage_matrix, annot=True,  fmt='.2f', xticklabels=['0','1','2','3','4','5','6','7','8','9'], yticklabels=['0','1','2','3','4','5','6','7','8','9']);
#     plt.title('Confusion Matrix');
#     m = a
# display_cm(a)

