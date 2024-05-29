import time
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorAssembler

#Calculate Execution time
start_time = time.time()

# SparkSession
spark = SparkSession.builder \
    .appName("RandomForestClassifierExample") \
    .getOrCreate()

# Define struct for creditcard dataset
schema = StructType([
    StructField("V1", DoubleType(), False),
    StructField("V2", DoubleType(), False),
    StructField("V3", DoubleType(), False),
    StructField("V4", DoubleType(), False),
    StructField("V5", DoubleType(), False),
    StructField("V6", DoubleType(), False),
    StructField("V7", DoubleType(), False),
    StructField("V8", DoubleType(), False),
    StructField("V9", DoubleType(), False),
    StructField("V10", DoubleType(), False),
    StructField("V11", DoubleType(), False),
    StructField("V12", DoubleType(), False),
    StructField("V13", DoubleType(), False),
    StructField("V14", DoubleType(), False),
    StructField("V15", DoubleType(), False),
    StructField("V16", DoubleType(), False),
    StructField("V17", DoubleType(), False),
    StructField("V18", DoubleType(), False),
    StructField("V19", DoubleType(), False),
    StructField("V20", DoubleType(), False),
    StructField("V21", DoubleType(), False),
    StructField("V22", DoubleType(), False),
    StructField("V23", DoubleType(), False),
    StructField("V24", DoubleType(), False),
    StructField("V25", DoubleType(), False),
    StructField("V26", DoubleType(), False),
    StructField("V27", DoubleType(), False),
    StructField("V28", DoubleType(), False),
    StructField("Amount", DoubleType(), False),
    StructField("Class", DoubleType(), False)
])

# Read from csv
data = spark.read.schema(schema).csv("/src/dataset/creditcard.csv")

# Remove na value
data = data.na.drop()

# LabelIndexer
labelIndexer = StringIndexer(inputCol="Class", outputCol="indexedLabel").fit(data)

# VectorAssembler for Feature
assembler = VectorAssembler(
    inputCols=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"],
    outputCol="features")

# Random split for training and test
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Create Model
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=10)

# Convert Label back to original
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)

# Pipeline
pipeline = Pipeline(stages=[labelIndexer, assembler, rf, labelConverter])

# Training
model = pipeline.fit(trainingData)

# Testing
predictions = model.transform(testData)

# Calculate error
evaluator = BinaryClassificationEvaluator(labelCol="indexedLabel", rawPredictionCol="prediction")

accuracy = evaluator.evaluate(predictions)

#Print Results
print("Test Error = %g" % (1.0 - accuracy))
print("--- %s seconds ---" % (time.time() - start_time))

spark.stop()
