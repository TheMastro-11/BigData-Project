from pyspark.ml import Pipeline
from sklearn.datasets import fetch_20newsgroups
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
import pandas as pd
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
 
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test= fetch_20newsgroups(subset='test')

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
    sc = SparkContext(appName="BigdataProject")
    spark = SparkSession(sc)

    train_set = preTextProcessing(200, newsgroups_train)
    test_set = preTextProcessing(40, newsgroups_test)

    # Crea un DataFrame con i dati di addestramento e di test
    df_train = spark.createDataFrame(pd.DataFrame(train_set))    
    df_test = spark.createDataFrame(pd.DataFrame(test_set))
 
    # Indicizza le etichette
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(df_train)
    
    # Esegui la tokenizzazione
    tokenizer = Tokenizer(inputCol="text", outputCol="words")

    # Calcola il TF
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")

    # Calcola l'IDF
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    # Crea un classificatore Random Forest
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features")

    # Crea un pipeline
    pipeline = Pipeline(stages=[labelIndexer, tokenizer, hashingTF, idf, rf])

    # Addestra il modello
    model = pipeline.fit(df_train)

    # Fai delle previsioni sul set di test
    predictions = model.transform(df_test)

    # Visualizza le previsioni
    predictions.select("prediction", "indexedLabel", "features").show(5)
    
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    rfModel = model.stages[4]
    print(rfModel)  # summary only 
    