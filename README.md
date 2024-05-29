# Goal
This project aims to test the timing and accuracy of different machine learning and deep learning algorithms using the multi-language engine Apache Spark.
It has been realised in collaboration with Prof. Reforgiato for the BigData and AdvancedBigDataArchitectures courses at the University of Cagliari.

# Requirements
[Spark (optional Hadoop)](https://github.com/TheMastro-11/spark-hadoop_configuration), Python3 and [dedicated libraries](requirements.sh) required.

# Models and Datasets
We used three different models, two of machine learning and one of deep learning. <br>
The [*creditcard*](/src/dataset/creditcard.csv) dataset is from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), *please download the full dataset from link as the version in this repository is partial*.


# Machine
The machine we used for these tests has the following characteristics:
* Intel® Core™ ¡5-6500 CPU @ 3.20GHz × 4
* 16,0 GiB
* Ubuntu 22.04.4 LTS

# Spark Configuration
Spark-submit allows you to add several parameters to customise the configuration, these are the ones we use:
* --master spark://master:7077
* --executor-memory 8G
* --total-executor-cores 4

# Results:
Repeated tests yielded the following results:
1. RandomForest: 
    * Test Error = 0.36113
    * 7,6 minuti

2. LogisticRegression:
    * Test Error = 0.233181
    * 28 secondi

3. KerasModel:
    * Test Error = 0.111258
    * 24.12903928756714 secondi

These results show both the superiority of a deep learning model over a classical one, and the importance of choosing the right model when making a prediction, with the Random Forest taking 18.24 times longer than the other two and also achieving a higher error rate.

#### Disclamair
Tests run in client mode as Spark does not support Python files for cluster mode.