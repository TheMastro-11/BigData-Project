from pyspark.ml import Pipeline
from sklearn.datasets import fetch_20newsgroups
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import RandomForestClassifier
import pandas as pd
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
 
categories_ = ["rec.autos", "rec.sport.baseball", "sci.electronics", "sci.med", "sci.space"]
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories_)
newsgroups_test= fetch_20newsgroups(subset='test', categories=categories_)

def preTextProcessing(amount, dataset):
    documents_ = {}
    documents_["label"] = []
    documents_["text"] = []
    for i in range(amount):
        #get metadata
        target = dataset.target[i]
        
        documents_["label"].append(target)
        documents_["text"].append(dataset.data[i]) #store in dictionary
        
    return documents_
    
    
if __name__ == "__main__":
    # Crea un SparkContext
    start_time = time.time()
    sc = SparkContext(appName="BigdataProject")
    spark = SparkSession(sc)

    train_set = preTextProcessing(len(newsgroups_test.data), newsgroups_train)
    test_set = preTextProcessing(len(newsgroups_test.data),newsgroups_test)

    # Crea un DataFrame con i dati di addestramento e di test
    df_train = spark.createDataFrame(pd.DataFrame(train_set))    
    df_test = spark.createDataFrame(pd.DataFrame(test_set))

    # Esegui la tokenizzazione
    tokenizer = Tokenizer(inputCol="text", outputCol="words")

    # Calcola il TF
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")

    # Calcola l'IDF
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    # Crea un classificatore Random Forest
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")

    # Crea un pipeline
    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, rf])
    
    # Addestra il modello
    model = pipeline.fit(df_train)

    # Fai delle previsioni sul set di test
    predictions = model.transform(df_test)
    
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))
    print("--- %s seconds ---" % (time.time() - start_time))

    