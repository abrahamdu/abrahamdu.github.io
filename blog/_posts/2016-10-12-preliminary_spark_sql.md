---
layout: post
section-type: post
published: true
category: BigData
tags: [ 'spark' ]
title: 'Preliminary Spark SQL'
date: 2016-10-12
modifiedOn: 2016-10-12
---

This following records my test of Spark SQL on Jupyter notebook.

__Step 1 - Working with Spark Context__  
---------------------------------------  
Invoke the spark context: sc. The version method will return the working version of Apache Spark.  

```python  
#Step 1 - Check spark version
sc.version 
#u'1.6.0'  
```  

__Step 2 - Working with Resilient Distributed Datasets__  
--------------------------------------------------------  

```python  
#Step 2.1 - Create RDD of numbers 1-10
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
x_nbr_rdd = sc.parallelize(x)
```
```python
#Step 2.2 - Return first element
x_nbr_rdd.first()
#1

#Step 2.3 - Return an array of the first five elements
x_nbr_rdd.take(5)
#[1, 2, 3, 4, 5]
```  

Perform a map transformation to increment each element of the array. The map function creates a new RDD by applying the function provided in the argument to each element.  

```python  
# Step 2.4 - Write your map function
x_nbr_rdd_2=x_nbr_rdd.map(lambda x: x+1)  #It's not required to be x.  (lambda a: a+1) would also work
x_nbr_rdd_2
# PythonRDD[5] at RDD at PythonRDD.scala:43
```

Note that there was no result for step 2.4. Why was this? Take a look at all the elements of the new RDD.  

```python
#Step 2.5 - Check out the elements of the new RDD. Warning: Be careful with this in real life! Collect returns everything!
x_nbr_rdd_2.collect()
#[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
x_nbr_rdd_2.count()
#10
```

```python
#Step 2.7 - Create String RDD with many lines / entries, Extract first line
z = 'IBM Data Science Experience is built for enterprise-scale deployment.', "Manage your data, your analytical assets, and your projects in a secured cloud environment." , "When you create an account in the IBM Data Science Experience, we deploy for you a Spark as a Service instance to power your analysis and 5 GB of IBM Object Storage to store your data."
z_str_rdd = sc.parallelize(z)
z_str_rdd.first()
#'IBM Data Science Experience is built for enterprise-scale deployment.'

#Step 2.9 - Show all the entries in the RDD
z_str_rdd.collect()
#['IBM Data Science Experience is built for enterprise-scale deployment.',
# 'Manage your data, your analytical assets, and your projects in a secured cloud #environment.',
# 'When you create an account in the IBM Data Science Experience, we deploy for #you a Spark as a Service instance to power your analysis and 5 GB of IBM Object #Storage to store your data.']
```

```python
#Step 2.10 - Perform a map transformation to split all entries in the RDD
#Check out the entries in the new RDD
z_str_rdd_split = z_str_rdd.map(lambda line: line.split(" "))
z_str_rdd_split.collect()
#[['IBM',
  'Data',
  'Science',
  'Experience',
  'is',
  'built',
  'for',
  'enterprise-scale',
  'deployment.'],
 ['Manage',
  'your',
  'data,',
  'your',
  'analytical',
  'assets,',
  'and',
  'your',
  'projects',
  'in',
  'a',
  'secured',
  'cloud',
  'environment.'],
 ['When',
  'you',
  'create',
  'an',
  'account',
  'in',
  'the',
  'IBM',
  'Data',
  'Science',
  'Experience,',
  'we',
  'deploy',
  'for',
  'you',
  'a',
  'Spark',
  'as',
  'a',
  'Service',
  'instance',
  'to',
  'power',
  'your',
  'analysis',
  'and',
  '5',
  'GB',
  'of',
  'IBM',
  'Object',
  'Storage',
  'to',
  'store',
  'your',
  'data.']]
```  

Creat a SQL context:  
```python
#Imports the SparkSQL library and start the connection to Spark
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

#Download the world bank file
!wget https://raw.githubusercontent.com/bradenrc/sparksql_pot/master/world_bank.json.gz
```  

Create a DataFrame:  
```python
#create the Dataframe here:
WBdf = sqlContext.read.json("world_bank.json.gz")

#print out the schema
WBdf.printSchema()
```

Register a table:  
```python
#Create the table to be referenced via SparkSQL
WBdf.registerTempTable('world_bank')
```

```python
#Use SQL to select from table limit 2 and print the output
query = """select * from world_bank limit 2"""
sqlContext.sql(query)

#Extra credit, take the DataFrame you created with the two records and convert it into a Pandas DataFrame
sqlContext.sql(query).toPandas()

#Now calculate a simple count based on a group, for example "regionname"
query2 = 'select count(*) as Count, regionname from world_bank group by regionname'
sqlContext.sql(query2).toPandas()

#Now calculate a simple count based on a group, for example "regionname"
query2 = 'select count(*) as Count, regionname from world_bank group by regionname'
sqlContext.sql(query2).toPandas()


```
