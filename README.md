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
