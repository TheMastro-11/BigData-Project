import time
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

# Calculate Execution time
start_time = time.time()

# SparkSession
spark = SparkSession.builder.appName("KerasEvaluationExample").getOrCreate()

# Define schema for credit card dataset
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
data = spark.read.schema(schema).csv("creditcard.csv")

# Remove na values
data = data.na.drop()

# Verify and ensure labels are binary (0 or 1)
data = data.withColumn("Class", col("Class").cast("double"))
data = data.filter((col("Class") == 0) | (col("Class") == 1))

# Create a features column
assembler = VectorAssembler(inputCols=[f"V{i}" for i in range(1, 29)] + ["Amount"], outputCol="features")
data = assembler.transform(data)

# Split the data into training and testing datasets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Convert Spark DataFrame to TensorFlow data
X_train = train_data.select("features").rdd.map(lambda row: row[0].toArray()).collect()  # Extract features as NumPy array
y_train = train_data.select("Class").rdd.map(lambda row: row[0]).collect()  # Extract labels as NumPy array

# Convert lists to NumPy arrays
import numpy as np
X_train = np.array(X_train)
y_train = np.array(y_train)

# Deep Learning Classifier with TensorFlow/Keras
model = Sequential([
    Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer with 128 neurons and ReLU activation
    Dense(units=64, activation='relu'),                           # Second hidden layer with 64 neurons and ReLU activation
    Dense(units=1, activation='sigmoid')                           # Output layer with 1 neuron and sigmoid activation for binary classification
])

# Model compilation
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model training
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Convert Spark DataFrame to TensorFlow data
X_test = test_data.select("features").rdd.map(lambda row: row[0].toArray()).collect()  # Extract features as NumPy array
y_test = test_data.select("Class").rdd.map(lambda row: row[0]).collect()  # Extract labels as NumPy array

# Convert lists to NumPy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Error = %g" % (1.0 - accuracy))
print("--- %s seconds ---" % (time.time() - start_time))

spark.stop()