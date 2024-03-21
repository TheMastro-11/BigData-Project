from pyspark.ml import Pipeline
from sklearn.datasets import fetch_20newsgroups
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import RandomForestClassifier
import pandas as pd
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
 
categories_ = ["rec.autos", "rec.sport.baseball", "sci.electronics", "sci.med", "sci.space"] 
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories_)
newsgroups_test= fetch_20newsgroups(subset='test', categories=categories_)

def preTextProcessing(amount, dataset):
    documents_ = {}
    documents_["target"] = []
    documents_["text"] = []
    for i in range(amount):
        #get metadata
        target = dataset.target[i]
        words = dataset.data[i].split(" ")
        
        documents_["target"].append(target)
        documents_["text"].append(words) #store in list
        
    return documents_
    
    
if __name__ == "__main__":
    # Crea un SparkContext
    sc = SparkContext(appName="BigdataProject")
    spark = SparkSession(sc)

    train_set = preTextProcessing(len(newsgroups_train.data), newsgroups_train)
    test_set = preTextProcessing(len(newsgroups_test.data),newsgroups_test)

    # Crea un DataFrame con i dati di addestramento e di test
    df_train = spark.createDataFrame(pd.DataFrame(train_set))    
    df_test = spark.createDataFrame(pd.DataFrame(test_set))
    
    #Vectorizer
    cv = CountVectorizer(inputCol="text", outputCol="vectors")

    # Crea un classificatore Random Forest
    rf = RandomForestClassifier(labelCol="target", featuresCol="vectors")

    # Crea un pipeline
    pipeline = Pipeline(stages=[cv, rf])
    
    # Addestra il modello
    model = pipeline.fit(df_train)

    # Fai delle previsioni sul set di test
    predictions = model.transform(df_test)
    
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="target", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    