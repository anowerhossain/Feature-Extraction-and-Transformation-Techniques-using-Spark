# Feature-Extraction-and-Transformation-Techniques-using-Spark

## Task 1 - Tokenizer 🧩
- A Tokenizer is used to split a sentence into words. This is a fundamental step in natural language processing (NLP) to break down sentences into their components.

```python
from pyspark.ml.feature import Tokenizer

# Sample DataFrame
sentenceDataFrame = spark.createDataFrame([
    (1, "Spark is a distributed computing system."),
    (2, "It provides interfaces for multiple languages"),
    (3, "Spark is built on top of Hadoop")
], ["id", "sentence"])

# Tokenizer
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
token_df = tokenizer.transform(sentenceDataFrame)

# Display Results
token_df.show(truncate=False)
```
Output:
Displays a column with tokenized words.



## Task 2 - CountVectorizer 🔢
- The CountVectorizer converts text into a numerical format by counting the occurrences of each word in a document.

```python
from pyspark.ml.feature import CountVectorizer

# Sample DataFrame
textdata = [(1, "I love Spark Spark provides Python API ".split()),
            (2, "I love Python Spark supports Python".split()),
            (3, "Spark solves the big problem of big data".split())]

textdata = spark.createDataFrame(textdata, ["id", "words"])

# CountVectorizer
cv = CountVectorizer(inputCol="words", outputCol="features")
model = cv.fit(textdata)
result = model.transform(textdata)

# Display Results
result.show(truncate=False)
```
Output:
Features column represents word counts for each document.



## Task 3 - TF-IDF 📊
- TF-IDF (Term Frequency-Inverse Document Frequency) quantifies the importance of words across documents.

```python
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

# Sample DataFrame
sentenceData = spark.createDataFrame([
    (1, "Spark supports python"),
    (2, "Spark is fast"),
    (3, "Spark is easy")
], ["id", "sentence"])

# Tokenization
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)

# HashingTF and IDF
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10)
featurizedData = hashingTF.transform(wordsData)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
tfidfData = idfModel.transform(featurizedData)

# Display Results
tfidfData.select("sentence", "features").show(truncate=False)

```

Output:
Features column contains the importance score of each word.


## Task 4 - StopWordsRemover 🛑
- StopWordsRemover filters out common words (like is, the) to improve text processing.

```python

from pyspark.ml.feature import StopWordsRemover

# Sample DataFrame
textData = spark.createDataFrame([
    (1, ['Spark', 'is', 'an', 'open-source', 'distributed', 'computing', 'system']),
    (2, ['IT', 'has', 'interfaces', 'for', 'multiple', 'languages']),
    (3, ['It', 'has', 'a', 'wide', 'range', 'of', 'libraries', 'and', 'APIs'])
], ["id", "sentence"])

# StopWordsRemover
remover = StopWordsRemover(inputCol="sentence", outputCol="filtered_sentence")
textData = remover.transform(textData)

# Display Results
textData.show(truncate=False)

```

Output:
Filtered sentences with stop words removed.


## Task 5 - StringIndexer 🏷️
- StringIndexer converts categorical data into numeric indices.

```python

from pyspark.ml.feature import StringIndexer

# Sample DataFrame
colors = spark.createDataFrame(
    [(0, "red"), (1, "red"), (2, "blue"), (3, "yellow" ), (4, "yellow"), (5, "yellow")],
    ["id", "color"])

# StringIndexer
indexer = StringIndexer(inputCol="color", outputCol="colorIndex")
indexed = indexer.fit(colors).transform(colors)

# Display Results
indexed.show()

```

Output:
Numeric indices assigned to each category.


## Task 6 - StandardScaler ⚖️
- StandardScaler standardizes numerical data to have a mean of 0 and a standard deviation of 1.

```python

from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors

# Sample DataFrame
data = [(1, Vectors.dense([70, 170, 17])),
        (2, Vectors.dense([80, 165, 25])),
        (3, Vectors.dense([65, 150, 135]))]
df = spark.createDataFrame(data, ["id", "features"])

# StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scalerModel = scaler.fit(df)
scaledData = scalerModel.transform(df)

# Display Results
scaledData.show(truncate=False)

```
Output:
Scaled numerical features with standardized values.
