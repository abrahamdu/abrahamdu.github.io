---
layout: post
section-type: post
published: true
category: BigData
tags: [ 'bigdata' ]
title: 'Big Data Terminology'
date: 2016-09-12
modifiedOn: 2017-06-06
---

__Accumulo__ - A computer software project that developed a sorted, distributed key/value store based on the BigTable technology from Google. It is a system built on top of Apache Hadoop, Apache ZooKeeper, and Apache Thrift. Written in Java, Accumulo has cell-level access labels and server-side programming mechanisms.   

__Amazon EC2__ - Amazon Elastic Compute Cloud, is a web service that provides resizable compute capacity in the cloud. It is designed to make web-scale cloud computing easier for developers.    

__Amazon EC2 Container Service (ECS)__ - A highly scalable, high performance container management service that supports Docker containers and allows you to easily run applications on a managed cluster of Amazon EC2 instances.   

__Amazon S3__ - Amazon Simple Storage Service, provides developers and IT teams with secure, durable, highly-scalable cloud storage.  
 
__Avro__ - A remote procedure call and data serialization framework developed within Apache's Hadoop project. It uses JSON for defining data types and protocols, and serializes data in a compact binary format.    

__AWS__ - Amazon Web Services, a suite of cloud-computing services that make up an on-demand computing platform.  

__Cassandra__ - A free and open-source distributed database management system designed to handle large amounts of data across many commodity servers, providing high availability with no single point of failure. It offers robust support for clusters spanning multiple datacenters,[1] with asynchronous masterless replication allowing low latency operations for all clients.  

__Chukwa__ - An open source data collection system for monitoring large distributed systems. Chukwa is built on top of the Hadoop Distributed File System (HDFS) and Map/Reduce framework and inherits Hadoop’s scalability and robustness. Chukwa also includes a ﬂexible and powerful toolkit for displaying, monitoring and analyzing results to make the best use of the collected data.  

__Cypher__ - A declarative, SQL-inspired language for describing patterns in graphs which allows us to state what we want to select, insert, update or delete from our graph data without requiring us to describe exactly how to do it.  

__DAG__ - Directed Acyclic Graph. When a SparkContext is created, it is submitted to DAGScheduler. A stage is comprised of tasks based on partitions of the input data. The DAG scheduler pipelines operators together. The Stages are passed on to the Task Scheduler. The task scheduler launches tasks via cluster manager. And finally the Worker executes the tasks on the Slave.  

__Drill__ - An open-source software framework that supports data-intensive distributed applications for interactive analysis of large-scale datasets. It supports a variety of NoSQL databases and file systems, including HBase, MongoDB, MapR-DB, HDFS, MapR-FS, Amazon S3, Azure Blob Storage, Google Cloud Storage, Swift, NAS and local files. A single query can join data from multiple datastores.    

__Flink__ - An open source platform for distributed stream and batch data processing. The core of it is a streaming dataflow engine that provides data distribution, communication, and fault tolerance for distributed computations over data streams.    

__Flume__ - A distributed, reliable, and available service for efficiently collecting, aggregating, and moving large amounts of log data. It has a simple and flexible architecture based on streaming data flows. It is robust and fault tolerant with tunable reliability mechanisms and many failover and recovery mechanisms. It uses a simple extensible data model that allows for online analytic application.  

__Graph Database__ A database that uses graph structures for semantic queries with nodes, edges and properties to represent and store data. A key concept of the system is the graph (or edge or relationship), which directly relates data items in the store.   

__Hadoop__ - The Apache Hadoop software library is a framework that allows for the distributed processing of large data sets across clusters of computers using simple programming models. It is designed to scale up from single servers to thousands of machines, each offering local computation and storage.  

__HBase__ - A column-oriented database management system that runs on top of HDFS.    

__HDFS__ - A Java-based file system that provides scalable and reliable data storage, and it was designed to span large clusters of commodity servers.  

__Hive__ - A data warehouse infrastructure built on top of Hadoop for providing data summarization, query, and analysis. It gives an SQL-like interface to query data stored in various databases and file systems that integrate with Hadoop.  

__Kafka__ - A distributed, partitioned, replicated commit log service. It provides the functionality of a messaging system.  

__Lambda Architecture__ - A data-processing architecture designed to handle massive quantities of data by taking advantage of both batch- and stream-processing methods. This approach to architecture attempts to balance latency, throughput, and fault-tolerance by using batch processing to provide comprehensive and accurate views of batch data, while simultaneously using real-time stream processing to provide views of online data.  

__Mahout__ - A project of the Apache Software Foundation to produce free implementations of distributed or otherwise scalable machine learning algorithms focused primarily in the areas of collaborative filtering, clustering and classification.   

__MapReduce__ - A programming model and an associated implementation for processing and generating large data sets with a parallel, distributed algorithm on a cluster.  

__Mesos__ - It is built using the same principles as the Linux kernel, only at a different level of abstraction. The Mesos kernel runs on every machine and provides applications (e.g., Hadoop, Spark, Kafka, Elasticsearch) with API’s for
resource management and scheduling across entire datacenter and cloud environments.  

__MongoDB__ - A cross-platform, open-source database that uses a document-oriented data model, rather than a traditional table-based relational database structure. This type of database structure is designed to make the integration of structured and unstructured data in certain types of applications easier and faster.  

__Oozie__ - A Java Web application used to schedule Apache Hadoop jobs. It is integrated with the rest of the Hadoop stack supporting several types of Hadoop jobs out of the box (such as Java map-reduce, Streaming map-reduce, Pig, Hive, Sqoop and Distcp) as well as system specific jobs (such as Java programs and shell scripts).    

__Parquet__ - A columnar storage format available to any project in the Hadoop ecosystem, regardless of the choice of data processing framework, data model or programming language.  

__Pig__ - A platform for analyzing large data sets that consists of a high-level language for expressing data analysis programs, coupled with infrastructure for evaluating these programs. The salient property of Pig programs is that their structure is amenable to substantial parallelization, which in turns enables them to handle very large data sets.  

__Redis__ - An open source (BSD licensed), in-memory data structure store, used as database, cache and message broker. It supports data structures such as strings, hashes, lists, sets, sorted sets with range queries, bitmaps, hyperloglogs and geospatial indexes with radius queries.  

__REST__ - REpresentational State Transfer, an architectural style, and an approach to communications that is often used in the development of Web services.  

__Shark__ - Also known as SQL on Spark, is a large-scale data warehouse system for Spark designed to be compatible with Apache Hive. Shark supports Hive's query language, metastore, serialization formats, and user-defined functions, providing seamless integration with existing Hive deployments and a familiar, more powerful option for new ones. It has been subsumed by Spark SQL.     

__Solr__ - A highly reliable, scalable and fault tolerant, providing distributed indexing, replication and load-balanced querying, automated failover and recovery, centralized configuration and more.    

__Spark__ - A fast, in-memory data processing engine with elegant and expressive development APIs to allow data workers to efficiently execute streaming, machine learning or SQL workloads that require fast iterative access to datasets.     

__Sqoop__ - A tool designed for efficiently transferring bulk data between Apache Hadoop and structured datastores such as relational databases.  

__Storm__ - A distributed real-time computation system for processing large volumes of high-velocity data. It is extremely fast, with the ability to process over a million records per second per node on a cluster of modest size.  

__TensorFlow__ - An open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. This flexible architecture lets you deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without rewriting code. TensorFlow also includes TensorBoard, a data visualization toolkit. It was originally developed and maintained by Google.  

__Tez__ - An extensible framework for building high performance batch and interactive data processing applications, coordinated by YARN in Apache Hadoop. It improves the MapReduce paradigm by dramatically improving its speed, while maintaining MapReduce’s ability to scale to petabytes of data.  

__Thrift__ - An interface definition language and binary communication protocol that is used to define and create services for numerous languages. It is used as a remote procedure call (RPC) framework and was developed at Facebook for "scalable cross-language services development".  

__YARN__ - Yet Another Resource Negotiator, is the architectural center of Hadoop that allows multiple data processing engines such as interactive SQL, real-time streaming, data science and batch processing to handle data stored in a single platform, unlocking an entirely new approach to analytics.  

__ZooKeeper__ - A centralized service for maintaining configuration information, naming, providing distributed synchronization, and providing group services.  