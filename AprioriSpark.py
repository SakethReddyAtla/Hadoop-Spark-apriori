from pyspark.sql import SparkSession
from pyspark.sql.functions import split,collect_set, regexp_replace, explode, collect_list, size
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Apriori") \
    .getOrCreate()

# Read the CSV file
# df = spark.read.option("header", "true").csv("hdfs://localhost:9000/SakethInput/outputmap.csv")


df = spark.read.option("delimiter", "\t").csv("hdfs://localhost:9000/SakethInput/outputmap.csv").toDF("transaction_id", "itemset")

# Split the itemset column
df = df.withColumn("items", split(df["itemset"], ","))

# Explode the items column to get individual items
df = df.withColumn("item", explode(df["items"]))

# Select only the necessary columns
df = df.select("transaction_id", "item")

# Show the DataFrame
df.show(truncate=False)


transactions = df.groupBy("transaction_id").agg(collect_set("item").alias("items"))

# Apply FP-Growth algorithm
fpGrowth = FPGrowth(itemsCol="items", minSupport=0.02, minConfidence=0.02)
model = fpGrowth.fit(transactions)


#Second research question

# Display frequent itemsets
print("Frequent itemsets:")
model.freqItemsets.show()

# Display association rules
print("Association rules:")
model.associationRules.show()


frequent_itemsets = model.freqItemsets.toPandas()
association_rules = model.associationRules.toPandas()



frequent_itemsets.to_csv("frequent_itemsets.csv", index=False)
association_rules.to_csv("association_rules.csv", index=False)


#Third research question
recommendations = model.transform(transactions)

recommendations.show()

association_rules['consequent'] = association_rules['consequent'].apply(lambda x: ', '.join(x))

sns.barplot(x="consequent", y="confidence", data=association_rules)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()