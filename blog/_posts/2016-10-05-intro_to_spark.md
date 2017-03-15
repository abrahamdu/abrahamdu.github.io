---
layout: post
section-type: post
published: true
category: BigData
tags: [ 'spark' ]
title: 'Introduction to Spark'
date: 2016-10-05
modifiedOn: 2016-10-05
---

Apache [Spark] [1] is an open source in-memory cluster computing framework optimized for extremely fast and large scale data processing. It started from AMPLab at UC Berkeley by [Matei Zaharia](https://cs.stanford.edu/~matei/) in 2009 and became an Apache open source project in 2010 and initially released in May 2014.    

![alt text](/img/blog/spark.png)  

__1. Spark Eco-System__  
----------------------  
__Spark Core__  
Spark Core is the underlying general execution engine for spark platform that all other functionality is built upon. It provides In-Memory computing and referencing datasets in external storage systems.  

__Spark SQL__  
Spark SQL is a component on top of Spark Core that introduces a new data abstraction called SchemaRDD, which provides support for structured and semi-structured data.  

__Spark Streaming__
Spark Streaming leverages Spark Core's fast scheduling capability to perform streaming analytics. It ingests data in mini-batches and performs RDD (Resilient Distributed Datasets) transformations on those mini-batches of data.  

__Spark ML__  
ML is a distributed machine learning framework above Spark because of the distributed memory-based Spark architecture. It is, according to benchmarks, done by the MLlib developers against the Alternating Least Squares (ALS) implementations. Spark MLlib is nine times as fast as the Hadoop disk-based version of Apache Mahout (before Mahout gained a Spark interface).  

__GraphX__  
GraphX is a distributed graph-processing framework on top of Spark. It provides an API for expressing graph computation that can model the user-defined graphs by using Pregel abstraction API. It also provides an optimized runtime for this abstraction.  

Besides the core and key libraries, [Spark] [1] also provides APIs to Java, Scala, Python and R. It could also work on a large number of data platforms, such as HDFS, Canssandra, HBase.  

![alt text](/img/blog/spark_ecosystem.jpg)  

__2. Spark Workhorse__  
----------------------  
RDD, which is Resilient Distributed Dataset, is the first generation of [Spark] [1] API born in 2011. Basically, it distributes collection of JVM objects and holds references to partition objects. And each partition is a subset of the overall data and assigned to nodes on cluster. It is immutable and fault tolerance. You can apply transformation method to it, such as *map*, *filter*, *flatmap*. However, these methods are just defining the operations to be performed and transformation wouldn't be performed until action method is called, such as *collect*, *take* and *first*. This work route is called *Lazy evaluation*.    

In 2013 with [Spark] [1] 1.3 release, DataFrame API was introduced which is a distributed collection of row objects. It becomes the main data object abstraction of SparkSQL, organized into *NAMED* columns. It can be created from existed data source, such as json, database table or Parquet, or from another RDD, or transformation from another DataFrame.   

In 2015 with [Spark] [1] 1.6 release, a preview of the new Dataset API was introduced into [Spark] [1]. Essentially, DataSet is internally rows, and JVM objects externally and limited to classes that implement the Scala Product trait, such as case classes. Datasets store data in an optimized binary format, often in off-heap memory, to avoid the costs of deserialization and garbage collection. Even though it feels like you are coding against regular objects, Spark is really generating its own optimized byte-code for accessing the data directly.  

__3. Code Execution__  
---------------------  
In a distributed system like [Spark] [1], a conventional program wouldn't work  as data is split across nodes. DAG (Directed Acyclic Graph) is a programming style for distributed systems - You can think of it as an alternative to Map Reduce. While MR has just two steps (map and reduce), DAG can have multiple levels that can form a tree structure. Basically, when any action is called on the RDD, [Spark] [1] creates the DAG and submits it to the DAG scheduler. The DAG scheduler divides operators into stages of tasks. A stage is comprised of tasks based on partitions of the input data. The DAG scheduler pipelines operators together. For e.g. Many map operators can be scheduled in a single stage. The final result of a DAG scheduler is a set of stages. The Stages are passed on to the Task Scheduler.The task scheduler launches tasks via cluster manager (Spark Standalone/Yarn/Mesos). The task scheduler doesn't know about dependencies of the stages. The Worker executes the tasks on the Slave.   

![alt text](/img/blog/DAG.png)  

 

<br />

<u>Reference:</u>  
-----------------
(1). A Tale of Three Apache Spark APIs: RDDs, DataFrames, and Datasets, _[https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html](https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html)_.  
(2). Apache Spark 2.0 API Improvements: RDD, DataFrame, Dataset and SQL, _[http://www.agildata.com/apache-spark-2-0-api-improvements-rdd-dataframe-dataset-sql/](http://www.agildata.com/apache-spark-2-0-api-improvements-rdd-dataframe-dataset-sql/)_.  


[1]: http://spark.apache.org/  "Spark"