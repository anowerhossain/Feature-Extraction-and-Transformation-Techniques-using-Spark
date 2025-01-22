# Install necessary libraries (uncomment these lines if running in a new environment)
# !pip install pyspark==3.1.2 -q
# !pip install findspark -q

# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# Import FindSpark to initialize PySpark
import findspark
findspark.init()

# Import necessary PySpark modules
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand

# Create SparkSession
spark = SparkSession.builder.appName("Feature Extraction and Transformation using Spark").getOrCreate()

# Task 1: Tokenizer
# Tokenizer is used to split a sentence into individual words

from pyspark.ml.feature import Tokenizer

# Create a sample DataFrame with sentences
sentenceDataFrame = spark.createDataFrame([
    (1, "Spark is a distributed computing system."),
    (2, "It provides interfaces for multiple languages"),
    (3, "Spark is built on top of Hadoop")
], ["id", "sentence"])

# Display the original DataFrame
print("Original DataFrame:")
sentenceDataFrame.show(truncate=False)

# Create Tokenizer instance
# Input: "sentence" column, Output: "words" column
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

# Tokenize the data
token_df = tokenizer.transform(sentenceDataFrame)

# Display the tokenized DataFrame
print("Tokenized DataFrame:")
token_df.show(truncate=False)

# Task 2: CountVectorizer
# CountVectorizer converts text into a bag-of-words representation

from pyspark.ml.feature import CountVectorizer

# Create a sample DataFrame with tokenized words
textdata = [(1, "I love Spark Spark provides Python API ".split()),
            (2, "I love Python Spark supports Python".split()),
            (3, "Spark solves the big problem of big data".split())]

textdata = spark.createDataFrame(textdata, ["id", "words"])

# Display the original DataFrame
print("Text Data:")
textdata.show(truncate=False)

# Create a CountVectorizer instance
cv = CountVectorizer(inputCol="words", outputCol="features")

# Fit the model and transform the data
model = cv.fit(textdata)
result = model.transform(textdata)

# Display the DataFrame with CountVectorizer features
print("CountVectorizer Features:")
result.show(truncate=False)

# Task 3: TF-IDF
# TF-IDF quantifies the importance of a word in a document relative to the entire dataset

from pyspark.ml.feature import HashingTF, IDF, Tokenizer

# Create a sample DataFrame with sentences
sentenceData = spark.createDataFrame([
    (1, "Spark supports python"),
    (2, "Spark is fast"),
    (3, "Spark is easy")
], ["id", "sentence"])

# Display the original DataFrame
print("Sentence Data:")
sentenceData.show(truncate=False)

# Tokenize the sentences
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)

# Display the tokenized DataFrame
print("Tokenized Sentence Data:")
wordsData.show(truncate=False)

# Create a HashingTF instance
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10)
featurizedData = hashingTF.transform(wordsData)

# Display the hashed features
print("Hashed Features:")
featurizedData.show(truncate=False)

# Create an IDF instance
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
tfidfData = idfModel.transform(featurizedData)

# Display the DataFrame with TF-IDF features
print("TF-IDF Features:")
tfidfData.select("sentence", "features").show(truncate=False)

# Task 4: StopWordsRemover
# StopWordsRemover filters out common stop words like "a", "an", and "the"

from pyspark.ml.feature import StopWordsRemover

# Create a sample DataFrame with tokenized sentences
textData = spark.createDataFrame([
    (1, ['Spark', 'is', 'an', 'open-source', 'distributed', 'computing', 'system']),
    (2, ['IT', 'has', 'interfaces', 'for', 'multiple', 'languages']),
    (3, ['It', 'has', 'a', 'wide', 'range', 'of', 'libraries', 'and', 'APIs'])
], ["id", "sentence"])

# Display the original DataFrame
print("Text Data with Stop Words:")
textData.show(truncate=False)

# Remove stop words
remover = StopWordsRemover(inputCol="sentence", outputCol="filtered_sentence")
filteredData = remover.transform(textData)

# Display the DataFrame without stop words
print("Filtered Sentence Data:")
filteredData.show(truncate=False)

# Task 5: StringIndexer
# StringIndexer converts categorical strings into numeric indices

from pyspark.ml.feature import StringIndexer

# Create a sample DataFrame with categorical data
colors = spark.createDataFrame([
    (0, "red"),
    (1, "red"),
    (2, "blue"),
    (3, "yellow"),
    (4, "yellow"),
    (5, "yellow")
], ["id", "color"])

# Display the original DataFrame
print("Original Colors Data:")
colors.show()

# Index the color column
indexer = StringIndexer(inputCol="color", outputCol="colorIndex")
indexed = indexer.fit(colors).transform(colors)

# Display the indexed DataFrame
print("Indexed Colors Data:")
indexed.show()

# Task 6: StandardScaler
# StandardScaler scales features to have a mean of 0 and a standard deviation of 1

from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors

# Create a sample DataFrame with numerical features
data = [(1, Vectors.dense([70, 170, 17])),
        (2, Vectors.dense([80, 165, 25])),
        (3, Vectors.dense([65, 150, 135]))]

df = spark.createDataFrame(data, ["id", "features"])

# Display the original DataFrame
print("Original Numerical Features:")
df.show()

# Scale the features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scalerModel = scaler.fit(df)
scaledData = scalerModel.transform(df)

# Display the scaled DataFrame
print("Scaled Features:")
scaledData.show(truncate=False)

# Stop the Spark session
spark.stop()
