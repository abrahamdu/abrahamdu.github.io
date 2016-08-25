---
layout: post
section-type: post
published: true
category: TECH
tags: [ 'json', 'r', 'web' ]
title: 'Understand JSON - Part 3: Parse JSON with R'
date: 2016-03-10
modifiedOn: 2016-08-06
---

Besides the native support of JavaScript to parse [JSON] [1] string into object and other JavaScript Libraries' efforts (see a comprehensive list from [JSON] [1] although a little out-dated), there are also some other languages with support to [JSON] [1] data. In this section, I will go through [R] [0]'s three packages ("rjson", "RJSONIO" and "jsonlite") and how they work with [JSON] [1] data.

__0. Install [R] [0] Packages__  
-------------------------
Before digging into how each [R] [0] package manage [JSON] [1] objects, it is a prerequisite to install [R] [0] packages first.  

```r  
#### Install all three packages and their related packages.  

install.packages(c("rjson", "RJSONIO", "jsonlite", "tidyjson"), dependencies = TRUE)  

eg1 <- "[true,false,null]"
eg2 <- '{"a":true,"b":false,"c":null}'  
```   
I also create two [R] [0] vectors containing characters only to represent [JSON] [1] objects for following use.
                                                                                        
__1. [rjson] [2]__  
------------  
_[rjson] [2]_ was first implemented for [R] [0] in 2007 by _Alex Couture-Beil_. It allows [R] [0] users to convert [JSON] [1] objects into [R] [0] object and vice-verse. There are three functions available under _[rjson] [2]_ package: _fromJSON_, _toJSON_ and _newJSONParser_.

(1a). _fromJSON_ - From [JSON] [1] to [R] [0]  

```r
#### rjson
library(rjson)

a <- fromJSON( "[true, false, null]" )
a
## [[1]]
## [1] TRUE
## 
## [[2]]
## [1] FALSE
## 
## [[3]]
## NULL
 
class(a)
## [1] "list"

b <- fromJSON( '{"a":true, "b":false, "c":null}' )
b
## $a
## [1] TRUE
## 
## $b
## [1] FALSE
## 
## $c
## NULL

class(b)
## [1] "list"  
```

(1b). _toJSON_ - From [R] [0] to [JSON] [1]  

```r  
A <- toJSON(a)
A
## [1] "[true,false,null]"

class(A)
## [1] "character"

A == "[true, false, null]"
## [1] FALSE

B <- toJSON(b)
B
## [1] "{\"a\":true,\"b\":false,\"c\":null}"
cat(B)
## {"a":true,"b":false,"c":null}

class(B)
## [1] "character"

B == '{"a":true, "b":false, "c":null}'
## [1] FALSE  
```  

(1c). _newJSONParser_  
It is used to convert a collection of [JSON] [1] objects into [R] [0] objects.

```r  

```  

(1d). Methods Used for Implementation  

```r  
c <- toJSON(c(1:1e5))
system.time( C1 <- fromJSON(c,method = "C") )
## user  system elapsed 
## 0.05    0.00    0.05 

system.time( C2 <- fromJSON(c,method = "R") )
## user  system elapsed 
## 92.45    0.41   93.38   
```  

__2. [RJSONIO] [3]__  
-------------  
_[RJSONIO] [3]_ started with a GitHub project by [Duncan Temple Lang](http://www.stat.ucdavis.edu/~duncan/) in 2010. It also provides facilities for reading and writing data in [JSON] [1]. This allows [R] [0] objects to be inserted into JavaScript/ECMAScript/ActionScript code and allows [R] [0] programmers to read and convert [JSON] [1] content to [R] [0] objects. It could be used as an alternative to _[rjson] [2]_ package however it doesn't use S4/S3 methods. Given this, _[RJSONIO] [3]_ is extensible, allowing others to define S4 methods for different [R] [0] classes/types, as well as allowing the caller to specify a different callback handler. Unlike _[rjson] [2]_ package, _[RJSONIO] [3]_ package uses a C++ library - libjson, rather than implementing yet another [JSON] [1] parser so that parsing would be faster than pure interpreted [R] [0] code. There are three primary functions available in this package: fromJSON, toJSON and asJSVars.    
   
(2a). _fromJSON_ - Convert [JSON] [1] content to [R] [0] objects  

```r  
#### RJSONIO
library(RJSONIO)

a <- fromJSON( "[true, false, null]" )
a
## [[1]]
## [1] TRUE
## 
## [[2]]
## [1] FALSE
## 
## [[3]]
## NULL

class(a)
## [1] "list"

b <- fromJSON( '{"a":true, "b":false, "c":null}' )
b
## $a
## [1] TRUE
## 
## $b
## [1] FALSE
## 
## $c
## NULL

class(b)
## [1] "list"  
```   

(2b). _toJSON_ - Convert an [R] [0] object to a string in [JSON] [1]  

```r  
A <- toJSON(a)
A
## [1] "[true,false,null]"

class(A)
## [1] "character"

A == "[true, false, null]"
## [1] FALSE

B <- toJSON(b)
B
## [1] "{\"a\":true,\"b\":false,\"c\":null}"
cat(B)
## {"a":true,"b":false,"c":null}

class(B)
## [1] "character"

B == '{"a":true, "b":false, "c":null}'
## [1] FALSE  
```   

(2c). _asJSVars_ - Serialize [R] [0] objects as JavaScript/ECMAScript/ActionScript variables  

```r  
cat(asJSVars( a = 1:10, myMatrix = matrix(1:15, 3, 5),qualifier = "protected", types = TRUE))
## protected a : Array = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ] ;
##
## protected myMatrix : Array = [ [ 1, 4, 7, 10, 13 ],
##                              [ 2, 5, 8, 11, 14 ],
##                              [ 3, 6, 9, 12, 15 ] ] ;
```  

__3. [jsonlite] [4]__  
-------------  
_[jsonlite] [4]_ is commonly known to [R] [0] community starting from a 'fork' of _[RJSONIO] [3]_ package, starting from 2013 but has been completely rewritten in recent versions. Like _[RJSONIO] [3]_, it also provides functions, such as _fromJSON()_ and _toJSON()_ to convert between [JSON] [1] data and [R] [0] objects. It could also interact with web APIs, building pipelines and streaming data between [R] [0] and [JSON] [1].

(3a). _fromJSON_ and _toJSON_ - Convert Data between [R] [0] and [JSON] [1]  

```r
library(jsonlite)

jsonlite_a1 <- jsonlite::fromJSON(eg1)
jsonlite_a1
## [1]  TRUE FALSE    NA

class(jsonlite_a1)
## [1] "logical"
is(jsonlite_a1)
## [1] "logical" "vector"

jsonlite_a2 <- jsonlite::fromJSON(eg1, simplifyVector = F)
jsonlite_a2
## [[1]]
## [1] TRUE
##
## [[2]]
## [1] FALSE
## 
## [[3]]
## NULL

class(jsonlite_a2)
## [1] "list"
is(jsonlite_a2)
## [1] "list"   "vector"

jsonlite_b <- jsonlite::fromJSON(eg2)
jsonlite_b
## $a
## [1] TRUE
## 
## $b
## [1] FALSE
## 
## $c
## NULL  

class(jsonlite_b)
## [1] "list"
is(jsonlite_b)
## [1] "list"   "vector"  
```  

_[jsonlite] [4]_ provides more options in _fromJSON_ function. In the example above, _[jsonlite] [4]_ converts a [JSON] [1] array into a [R] [0] vector but after using arugments "simplifyVector = F", the [JSON] [1] array is converted into a [R] [0] string. However, if it is a [JSON] [1] object, the conversion will return a [R] [0] object automatically.

```r
jsonlite_A1 <- jsonlite::toJSON(jsonlite_a1)
jsonlite_A1
## [true,false,null]

class(jsonlite_A1)
## [1] "json"
is(jsonlite_A1)
## [1] "json"     "oldClass"

jsonlite_A2 <- jsonlite::toJSON(jsonlite_a2)
jsonlite_A2
## [[true],[false],{}] 

class(jsonlite_A2)
## [1] "json"
is(jsonlite_A2)
## [1] "json"     "oldClass"  

jsonlite_B <- jsonlite::toJSON(jsonlite_b)
jsonlite_B
## {"a":[true],"b":[false],"c":{}}

class(jsonlite_B)
## [1] "json"
is(jsonlite_B)
## [1] "json"     "oldClass"

jsonlite_B <- jsonlite::toJSON(jsonlite_b,null='null')
jsonlite_B
## {"a":[true],"b":[false],"c":null}

class(jsonlite_B)
## [1] "json"
is(jsonlite_B)
## [1] "json"     "oldClass"

jsonlite_B <- jsonlite::toJSON(jsonlite_b,null='list',pretty=T)
jsonlite_B
## {
##   "a": [true],
##   "b": [false],
##   "c": {}
## } 

class(jsonlite_B)
## [1] "json"
is(jsonlite_B)
## [1] "json"     "oldClass"
```  

_toJSON_ will convert [R] [0] object to [JSON] [1] and likewise, it provides more options for caller to make conversion explicitly, depending on the input class of [R] [0] object. It also allows the output to print in a 'pretty' way.  

(3b). _serializeJSON_ and _unserializeJSON_ - Convert Data between [R] [0] and [JSON] [1] Differently  

Away from the class-based encoding way by _fromJSON_ and _toJSON_ pairs, _serializeJSON_ and _unserializeJSON_ twins implement a type-based encoding to convert data between [R] [0] and [JSON] [1].  

```r
jsonlite_se_a1 <- jsonlite::serializeJSON(jsonlite_A1)
cat(jsonlite_se_a1)
## {"type":"character","attributes":{"class":{"type":"character","attributes":{},"value":["json"]}},"value":["[true,false,null]"]}

class(jsonlite_se_a1)
## [1] "character"

jsonlite_se_A1 <- jsonlite::unserializeJSON(jsonlite_se_a1)
jsonlite_se_A1
## [true,false,null]

class(jsonlite_se_a1)
## [1] "character"  

jsonlite_se_a2 <- jsonlite::serializeJSON(jsonlite_A2)
cat(jsonlite_se_a2)
## {"type":"character","attributes":{"class":{"type":"character","attributes":{},"value":["json"]}},"value":["[[true],[false],{}]"]}

class(jsonlite_se_a2)
## [1] "character"

jsonlite_se_A2 <- jsonlite::unserializeJSON(jsonlite_se_a2)
jsonlite_se_A2
## [[true],[false],{}] 

class(jsonlite_se_a2)
## [1] "character"
```  

In the examples above, using _serializeJSON_ could convert an [R] [0] object into [JSON] [1] and it captures the type, value and attributes of each storage type so the object can be restored almost perfectly from its [JSON] [1] representation. The cost of it is the lengthiness, sometimes redundancy of the result.  
 
Caveat: Besides the differences in encoding system between class-based (_fromJSON_ and _toJSON_) and type-based (_serializeJSON_ and _unserializeJSON_), there is another trivial difference when using them in terms of input file required for functions to work. _fromJSON_ and _toJSON_ are independent from each other so _fromJSON_ can use any [JSON] [1] file or the returned result from _toJSON_ as input and so can _toJSON_ do. However, _unserializeJSON_ has to use [JSON] [1] string created from _serializeJSON_ as input.    

```r
bad_unse <- jsonlite::unserializeJSON(eg1)
## Error in switch(encoding.mode, `NULL` = NULL, environment = new.env(parent = emptyenv()),  : 
##  EXPR must be a length 1 vector
```  

(3c). _stream_in_ and _stream_out_ - Streaming [JSON] [1] input and output  

As I mentioned before, [JSON] [1] file could carry huge amount of data from web and it becomes one of its advantages. However, since [R] [0] stores and processes all data in the memory, the power of [JSON] [1] is bounded by the limit of specific [R] [0] machines. In order to address this bottleneck, _jsonlite_ package implements these two functions to process data over a http(s) connection, a pipe, even from a NoSQL database. However different from _fromJSON_ and _toJSON_, the streaming requires the [ndjson](http://ndjson.org/) format.   
  
```r  
library(MASS)
stream_out(cats, stdout())
## {"Sex":"F","Bwt":2,"Hwt":7}
## {"Sex":"F","Bwt":2,"Hwt":7.4}
## {"Sex":"F","Bwt":2,"Hwt":9.5}
## {"Sex":"F","Bwt":2.1,"Hwt":7.2}
## {"Sex":"F","Bwt":2.1,"Hwt":7.3}
## ...

library(curl)
con <- curl("https://jeroenooms.github.io/data/diamonds.json")
mydata <- stream_in(con, pagesize = 1000)
## opening curl input connection.
## Imported 53940 records. Simplifying into dataframe...
## closing curl input connection.

head(mydata)
##   carat       cut color clarity depth table price    x    y    z
## 1  0.23     Ideal     E     SI2  61.5    55   326 3.95 3.98 2.43
## 2  0.21   Premium     E     SI1  59.8    61   326 3.89 3.84 2.31
## 3  0.23      Good     E     VS1  56.9    65   327 4.05 4.07 2.31
## 4  0.29   Premium     I     VS2  62.4    58   334 4.20 4.23 2.63
## 5  0.31      Good     J     SI2  63.3    58   335 4.34 4.35 2.75
```  

Besides the functionality of reading and writing data between [JSON] [1] and [R] [0] provided by all of these three packages, they all provide some other  different functions in each of them. For example, _jsonlite_ provides _base64_dec_ and _base64_enc_ to convert between raw vectors to text while the other two packages don't have this function. Validating strings in [JSON] [1] format is provided by _RJSONIO_ (_isJSONValid_ function) and _jsonlite_ (_validate_) while _rjson_ doesn't have. _jsonlite_ also provides the capability of re-formatting [JSON] [1] file into: 1). structure with indentation added from _prettify_, 2). file by removing all unnecessary indentation and white spaces which is actually adopted by a lot of JavaScript libraries. In terms of parsing results, [This paper](https://rstudio-pubs-static.s3.amazonaws.com/31702_9c22e3d1a0c44968a4a1f9656f1800ab.html) gives readers a brief comparison between three packages which is also worthy reading it.


<br />

<u>Reference:</u>  
-----------------

(1). _rjson_ reference manual, _[https://cran.r-project.org/web/packages/rjson/rjson.pdf](https://cran.r-project.org/web/packages/rjson/rjson.pdf)_.  
(2). _RJSONIO_ reference manual, _[https://cran.r-project.org/web/packages/RJSONIO/RJSONIO.pdf](https://cran.r-project.org/web/packages/RJSONIO/RJSONIO.pdf)_.  
(3). _jsonlite_ reference manual, _[https://cran.r-project.org/web/packages/jsonlite/jsonlite.pdf](https://cran.r-project.org/web/packages/jsonlite/jsonlite.pdf)_.  
(4). _tidyjson_ reference manual, _[https://cran.r-project.org/web/packages/tidyjson/tidyjson.pdf](https://cran.r-project.org/web/packages/tidyjson/tidyjson.pdf)_.  
(5). A biased comparsion of JSON packages in R, _[https://rstudio-pubs-static.s3.amazonaws.com/31702_9c22e3d1a0c44968a4a1f9656f1800ab.html](https://rstudio-pubs-static.s3.amazonaws.com/31702_9c22e3d1a0c44968a4a1f9656f1800ab.html)_.  
(6). Jeroen Ooms, 2014, The _jsonlite_ Package: A Practical and Consistent Mapping Between JSON Data and R Objects.  



[0]: https://www.r-project.org/  "R"
[1]: http://www.json.org/        "JSON"
[2]: https://cran.r-project.org/web/packages/rjson/index.html      "rjson"
[3]: https://cran.r-project.org/web/packages/RJSONIO/index.html    "RJSONIO"
[4]: https://cran.r-project.org/web/packages/jsonlite/index.html   "jsonlite"
[5]: https://cran.r-project.org/web/packages/tidyjson/index.html   "tidyjson"