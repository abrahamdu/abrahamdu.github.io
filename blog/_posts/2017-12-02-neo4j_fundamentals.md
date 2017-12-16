---
layout: post
section-type: post
published: true
category: BigData 
tags: [ 'neo4j' ]
title: 'Neo4j Fundamentals'
date: 2017-12-02
modifiedOn: 2017-12-02
---  

[Neo4j] [1] is a graph database which stores connections between nodes as first citizens. Different from traditional relational databases, such as [Oracle](https://www.oracle.com/database/index.html), a graph database infers from data connections rather than using keys to join different tables together. In other words, a graph database starts with nodes/connections to figure out its related neighborhood rather than using query on top of query to get the connected information. To better understand graph database, it is necessary to understand _Property Graph Model_ first.   

__1. Property Graph Model__
---------------------------
There are four basic elements in the _Property Graph Model_, _Nodes_ that represents the objects in the graph, _Relationships_ that relate nodes by type and direction, _Properties_ that are name-value pairs going with nodes and relationships, and _Labels_ that groups nodes by role. The property graph contains _nodes_ that hold any number of _properties_. Each _node_ could be tagged with different _labels_ to represent different roles in each domain. _Relationships_ provide directed relevant connections between different _nodes_ although they could also be accessed regardless directions. Like _nodes_, _relationships_ could also have any _properties_.     

__2. Cypher Query Language__
----------------------------  
_Cypher_ is a declarative, expressive and pattern matching query language for graphs by [Neo4j] [1]. t allows us to state what we want to select, insert, update or delete from our graph data without requiring us to describe exactly how to do it. _Cypher_ is using Ascii Art for Patterns. It is declarative, readable and expressive with powerful capabilities. It is also an open Language which you contribute to it through [GitHub](https://github.com/neo4j/neo4j).  

__Nodes__ is created by parentheses _( )_. It can be labeled by colon _:_ and can have properties by curly braces _{ }_ stored as key/value pairs. Properties can be strings, numbers, or booleans, and lists of strings, numbers, or booleans. For example:    

```sql  
CREATE (va:State {Name: "Virginia"}) // create a node with 'State' as its label and property in {}
CREATE (md:State {Name: "Maryland"}) // create another node

MATCH (n)
RETURN n
```  

__Relationships__ is created by hyphens _- -_ & square brackets _[ ]_ with arrows _< >_ to specify directions. Like nodes, relationships can have labels and properties.  

```sql  
MATCH (va:State),(md:State)
WHERE va.Name = "Virginia" and md.Name = "Maryland"
CREATE (va)-[:Bordered]->(md)

//Which is equivalent to
MERGE (va:State {Name: "Virginia"})
MERGE (md:State {Name: "Maryland"});
MERGE (va)-[:Bordered]->(md)

//Which is similar as
CREATE (va:State {Name: "Virginia"})<-[b:Bordered]-(md:State {Name: "Maryland"})

MATCH (n) 
RETURN n
```  
![alt text](/img/blog/Cypher1.png) 

The first query actually first finds the nodes and setup the relationship using _MATCH_ and _CREATE_. The third query is using the same command but it would create another node-relationship pair. If you don't want to create duplications, you should use _MERGE_ command since it is essentially a combination of find and create. However, using _MERGE_ may be quite expensive since it searches the whole existing relationships to figure out whether it needs to be created or not. Please note, in _Cypher_, __labels__, __relationship-types__, __property-names__ and __variables__ are case sensitive. All others are not.  

We can also update properties by using _SET_. For example, we can add capital city in each state node.  

```sql  
MATCH (va:State {Name: "Virginia"})
SET va.Capital = "Richmond"
RETURN va

MATCH (md:State {Name: "Maryland"})
SET md.Capital = "Annapolis"
RETURN md
```  
There are also some handy syntaxes to remember.  

```sql
//Showing all existing nodes/relationships
MATCH (n)
RETURN (n)

//Delete nodes
MATCH (va:State {Name: "Virginia", Capital: "Richmond"})
DETACH DELETE va

//Remove properties
MATCH (md:State {Name: "Maryland"})
REMOVE md.Capital RETURN md

//Delete relationships
MATCH (va:State {Name: "Virginia"})<-[b:Bordered]-(md:State {Name: "Maryland"})
DELETE b  

//Delete all
MATCH (n)
DELETE n
```  

__3. Reading Data from CSV__  
----------------------------  
Theoretically, nodes/relationships could be created by the methods above by saving the script with _.cyp_ as suffix. But you can imagine it won't scale once the data reach to a moderate size. In [Neo4j] [1], it uses __LOAD CSV__ as an ETL tool to load csv file from http(s) or file URL. The basic syntax is as follows:  

```sql  
[USING PERIODIC COMMIT] // optional transaction batching
LOAD CSV // load csv data
WITH HEADERS // optionally use first header row as keys in "row" map
FROM "url" // file:// URL relative to $NEO4J_HOME/import or http://
AS row // return each row of the CSV as list of strings or map
FIELDTERMINATOR ";" // alternative field delimiter 

... rest of the Cypher statement ...
```  

__4. From Relational to Graph Model__  
-------------------------------------  
You may be familiar with relational database by using _SQL_ to set up relationships between different tables while in [Neo4j] [1], we only see nodes and relationships. It is natural to think how to bridge the gap between relational model and graph model. Roughly speaking, you may want to think entity-tables as nodes and joins as relationships and all the foreign keys as relationships.  

<u>Reference:</u>    
----------------- 
(1). What is a Graph Database?, _[https://neo4j.com/developer/graph-database/](https://neo4j.com/developer/graph-database/)_.  
(2). Why Graph Databases?, _[https://neo4j.com/why-graph-databases/](https://neo4j.com/why-graph-databases/)_.  

[1]: https://neo4j.com/  "neo4j"