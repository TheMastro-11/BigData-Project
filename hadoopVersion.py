from sklearn.datasets import fetch_20newsgroups
from pyspark import SparkContext
from pyspark.sql import SparkSession

newsgroups_train = fetch_20newsgroups(subset='train')
#newsgroups_test= fetch_20newsgroups(subset='test')

amount = newsgroups_train.data

documents = {}

def preTextProcessing():
    documents_ = {}
    for i in range(amount):
        #get document
        currentDoc = newsgroups_train.data[i]
        currentDoc = currentDoc.lower().split() #lowercase - into words
        metadata = [] #list to store words before text
        for j in range(len(currentDoc)):
            metadata.append(currentDoc[j])
            if currentDoc[j] == "lines:":
                metadata.append(currentDoc[j+1])
                break
        textDoc = " ".join(currentDoc[len(metadata):]) #into string
        
        #get metadata
        name = newsgroups_train.filenames[i]
        id = name.split("/")[7]
        category = name.split("/")[6]
        
        sender = []
        for j in metadata:
            sender.append(j)
            if j == "subject:":
                sender.remove(j)
                break
        
        subject = []
        for j in metadata[len(sender):]:
            subject.append(j)
            if j == "nntp-posting-host:":
                subject.remove(j)
                break
        
        site = []
        for j in metadata[len(sender)+len(subject):]:
            site.append(j)
            if j == "organization:":
                site.remove(j)
                break
            
        organization = []
        for j in metadata[len(sender)+len(subject)+len(site):]:
            organization.append(j)
            if j == "lines:":
                organization.remove(j)
                break
            
        lines = []
        for j in metadata[len(sender)+len(subject)+len(site)+len(organization):]:
            lines.append(j)
        
        documents_[id] = category, " ".join(sender), " ".join(subject), " ".join(site), " ".join(organization), " ".join(lines), textDoc #store in dictionary
        
    return documents_
       

def load_dict_from_HDFS(path):
    """Load dictionary from HDFS
    
    Input
    -----
        path : str
    Return
    ------
        dict
    """
    return spark.read.load(path).collect()[0].asDict()

def save_dict_to_HDFS(dict_, path):
    """Store dictionary on HDFS

    Parameter:
    ----------
    dict_ : dict
        Dictionary to store on HDFS.
    path : str
    """
    sc.parallelize([dict_]).toDF().write.mode('overwrite').save(path)

if __name__ == "__main__":
    #Create a SparkContext
    sc = SparkContext(appName="BigdataProject")
    spark = SparkSession(sc)

    documents = preTextProcessing()
    '''
    new = {}
    j = 0
    for i in documents:
        new[i] = documents.get(i) 
        if j == 10:
            break
        j += 1
    '''

    #save dictionary on HDFS
    #save_dict_to_HDFS(documents, "hdfs://172.31.28.105:9000/prova")

    #load dictionary from HDFS
    #documents2 = load_dict_from_HDFS("hdfs://172.31.28.105:9000/prova")

    assert documents == documents2

    print("Test passed!")
    