---
layout: post
section-type: post
published: true
category: VISUALIZATION
tags: [ 'd3', 'javascript' ]
title: 'Read in Data by D3 - Part 2'
date: 2017-11-16
modifiedOn: 2017-11-24
---  

After introducing __d3-dsv__ module from [last blog](http://anotherpeak.org/blog/visualization/2017/10/26/read_in_data_d3_1.html), we learn how to use [D3] [1] to parse different types of flat files. To illustrate different charts [D3] [1] could achieve, a simple data set created during coding is good enough already (see [this](http://anotherpeak.org/blog/visualization/2017/07/02/d3_piechart.html)). But more often than not, it is more common to have an external data source residing in the local server or some remote server for [D3] [1]'s consumption. This blog aims to introduce using __d3-request__ module to pass different types of data files [D3] [1] could consume from the local server.  

__d3-request__ explicitly provides direct functions to be able to bring in the following types of files: html, xml, text, json, csv and tsv. We start with csv and tsv files since their structure is much simpler than others'. I used the data used in the [area chart](http://anotherpeak.org/blog/visualization/2017/06/28/d3_areachart.html) and [line chart](http://anotherpeak.org/blog/visualization/2017/06/08/d3_linechart.html) for illustration.   

__1. d3.csv__  
-------------
_d3.csv(url[[, row], callback])_ only requires a _url_ to return a new request. Usually, a _callback_ function is specified to pass the data object as a parameter to callback function once the data is loaded.  

```javascript  
<script src="https://d3js.org/d3-collection.v1.min.js"></script>
<script src="https://d3js.org/d3-dispatch.v1.min.js"></script>
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script src="https://d3js.org/d3-request.v1.min.js"></script>
<script>
  d3.csv("/blog/data/browser_statistics.csv",function(data){
    console.log(data);
//(118) [{…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, …]
//[0 … 99]
//[100 … 117]
//columns:(6) ["Date", "Chrome", "IE/Edge", "Firefox", "Safari", "Opera"]
//length: 118
//__proto__: Array(0)
    console.log(data[0]);
//{Date: "Oct-17", Chrome: "76.10", IE/Edge: "4.10", Firefox: "12.10", Safari: "3.30", …}
  });
</script>
```  
From the above example, using _d3.csv_ converts the raw csv file into an array of objects with a key/value pair. The header of csv file's column becomes the key and each row becomes value. And you may also notice the value is actually a string even though it should be numbers since they are the market share percentage in this example. This may not be what you actually look for. There are a few ways to address this. First, we can use the first optional parameter of _d3.csv_ function _row_ to map and filter row objectss to a more-specific representation,   

```javascript  
<script src="https://d3js.org/d3-collection.v1.min.js"></script>
<script src="https://d3js.org/d3-dispatch.v1.min.js"></script>
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script src="https://d3js.org/d3-request.v1.min.js"></script>
<script>
   d3.csv("/blog/data/browser_statistics.csv",function(data){  
    return{
        Date: data.Date,
        Chrome: +data.Chrome,
        "IE": +data["IE/Edge"],
        Firefox: +data.Firefox,
        Safari: +data.Safari,
        Opera: +data.Opera
        };    
  },function(data) {
  console.log(data[0]);
  });
</script>
//{Date: "Oct-17", Chrome: 76.1, IE: 4.1, Firefox: 12.1, Safari: 3.3, Opera: 1.2}
```  
which is also equivalent to:  

```javascript  
<script src="https://d3js.org/d3-collection.v1.min.js"></script>
<script src="https://d3js.org/d3-dispatch.v1.min.js"></script>
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script src="https://d3js.org/d3-request.v1.min.js"></script>
<script>
   d3.csv("/blog/data/browser_statistics.csv")
       .row(function(data){  
    return{
        Date: data.Date,
        Chrome: +data.Chrome,
        "IE": +data["IE/Edge"],
        Firefox: +data.Firefox,
        Safari: +data.Safari,
        Opera: +data.Opera
        };    
     })
   .get(function(data) {
  console.log(data[0]);
  });
</script>  
//{Date: "Oct-17", Chrome: 76.1, IE: 4.1, Firefox: 12.1, Safari: 3.3, Opera: 1.2}  
```  
In this way, you also have the full control over how the data is processed. For example, I modified the original "IE/Edge" with "IE".   

We can also just use the callback function alone and use _forEach_ to iterate over the whole array objects and combine it with _+_ to convert the original strings into numbers,    

```javascript  
<script src="https://d3js.org/d3-collection.v1.min.js"></script>
<script src="https://d3js.org/d3-dispatch.v1.min.js"></script>
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script src="https://d3js.org/d3-request.v1.min.js"></script>
<script>
   d3.csv("/blog/data/browser_statistics.csv",function(data){  
    data.forEach(function(d) {
    d.Chrome = +d.Chrome;
    d["IE/Edge"] = +d["IE/Edge"];
    d.Firefox = +d.Firefox;
    d.Safari = +d.Safari;
    d.Opera = +d.Opera;
  });
  console.log(data[0]);
  });
</script>  
//{Date: "Oct-17", Chrome: 76.1, IE/Edge: 4.1, Firefox: 12.1, Safari: 3.3, Opera: 1.2}
```  
which is equivalent to:  

```javascript  
<script src="https://d3js.org/d3-collection.v1.min.js"></script>
<script src="https://d3js.org/d3-dispatch.v1.min.js"></script>
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script src="https://d3js.org/d3-request.v1.min.js"></script>
<script>
   d3.csv("/blog/data/browser_statistics.csv")
       .get(function(data){  
    data.forEach(function(d) {
    d.Chrome = +d.Chrome;
    d["IE/Edge"] = +d["IE/Edge"];
    d.Firefox = +d.Firefox;
    d.Safari = +d.Safari;
    d.Opera = +d.Opera;
  });
  console.log(data[0]);
  });
</script>  
//{Date: "Oct-17", Chrome: 76.1, IE/Edge: 4.1, Firefox: 12.1, Safari: 3.3, Opera: 1.2}
```  

__2. d3.tsv__  
-------------
tsv file is the same as csv file except the delimiter is tab instead of comma. We can use the same method as _d3.csv_ by using _d3.tsv_ function instead. There is no need to repeat here.  

However, if the delimiter is something other than comma or tab, we can rely on __d3-dsv__ module to create a formatter first. And use __d3-request__ module to call _d3.request_, _mimeType_ and _response_ functions to pass the other delimited files to [D3] [1] as the example below shows:  

```javascript  
<script src="https://d3js.org/d3-collection.v1.min.js"></script>
<script src="https://d3js.org/d3-dispatch.v1.min.js"></script>
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script src="https://d3js.org/d3-request.v1.min.js"></script>
<script>
  var ssv = d3.dsvFormat(";");
  d3.request("/blog/data/age_by_gender.txt")
    .mimeType("text/plain")
    .response(function(xhr) { return ssv.parse(xhr.responseText) })
    .get(function(data) {
    console.log(data[0]);
  }); 
</script>  
//{Age_Group: "5 to 9 years", Male: "0.066", Female: "0.062"}
```  
As expected, the semi-colon delimited file is converted into an array of objects and all properties are strings. If this is not something we need, we can either pass _row_ parameter into _response_ function to convert properties into numbers:  

```javascript  
<script src="https://d3js.org/d3-collection.v1.min.js"></script>
<script src="https://d3js.org/d3-dispatch.v1.min.js"></script>
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script src="https://d3js.org/d3-request.v1.min.js"></script>
<script>
  var ssv = d3.dsvFormat(";");
  d3.request("/blog/data/age_by_gender.txt")
    .mimeType("text/plain")
    .response(function(xhr) { return ssv.parse(xhr.responseText, row) })
    .get(function(data) {
    console.log(data[0]);
  }); 
function row(d) {
  return {
    Age_Group: d.Age_Group,
    Male: +d.Male,
    Female: +d.Female
  };
}
</script>  
//{Age_Group: "Under 5 years", Male: 0.064, Female: 0.059}
```  
Or use _forEach_ to pass into callback function and iterate each object of array to convert into numbers:  

```javascript  
<script src="https://d3js.org/d3-collection.v1.min.js"></script>
<script src="https://d3js.org/d3-dispatch.v1.min.js"></script>
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script src="https://d3js.org/d3-request.v1.min.js"></script>
<script>
  var ssv = d3.dsvFormat(";");
  d3.request("/blog/data/age_by_gender.txt")
    .mimeType("text/plain")
    .response(function(xhr) { return ssv.parse(xhr.responseText) })
    .get(function(data) {
      data.forEach(function(d) {
        d.Male = +d.Male;
        d.Female = +d.Female;
    });
    console.log(data[0]);
  }); 
</script>  
//{Age_Group: "Under 5 years", Male: 0.064, Female: 0.059}
```  

__3. d3.text__  
--------------  
_d3.text_ only has two parameters, _url_ and _callback_ function. It is essentially the same as the example illustrated in _d3.tsv_ section with default mime type "text/plain". Nothing new.  

__4. d3.json__  
-------------- 
Like _d3.text_ function, _d3.json_ also has two parameters, _url_ and _callback_ function. It only differs in the mime type with default value "application/json". I use the json data set used in the [cluster diagram](http://anotherpeak.org/blog/visualization/2017/07/06/d3_clusterdiagram.html) as well as [tree diagram](http://anotherpeak.org/blog/visualization/2017/07/11/d3_treediagram.html) as example.  

```javascript  
<script src="https://d3js.org/d3-collection.v1.min.js"></script>
<script src="https://d3js.org/d3-dispatch.v1.min.js"></script>
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script src="https://d3js.org/d3-request.v1.min.js"></script>
<script>
   d3.json("/blog/data/all_countries_2015.json",function(data){
    console.log(data);
  });
</script>  
//{data: {…}, children: Array(6)}
```  
Again, we can see using _d3.json_ would convert the json file into an array of object and we can manipulate the array with the same method introduced in the section above.  

__5. d3.html and d3.xml__  
-------------------------  
Reading html and xml file is pretty much the same as reading json or text file. The key difference is that html has default mime type "text/html" and xml has default mime type "application/xml". The syntax and usage is quite similar to _d3.text_ or _d3.json_. I created a simple .xml file for illustration purposes.  

```javascript  
<script src="https://d3js.org/d3-collection.v1.min.js"></script>
<script src="https://d3js.org/d3-dispatch.v1.min.js"></script>
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script src="https://d3js.org/d3-request.v1.min.js"></script>
<script>
   d3.xml("/blog/data/example.xml",function(data){
    console.log(data);
    console.log(data.documentElement.getElementsByTagName("body"));
  });
</script>  
```  
__d3-request__ provides a full stack of parsing different types of data format for its consumption. Through different built-in functions introduced above, we could also generalize the commons between each other.  

In general, the six specific types of [D3] [1] function are just transformation of generic _d3.request().mimeType().response().get()_ calls which applies to different types of format. There are two kinds of plain files: text(csv, tsv, text & html) and application (json & xml). We just need to specify those types in _mimeType()_ function and pass it into _response()_ function. Here is a summary:  

```javascript  
////Read csv file
//Direct call:
d3.csv(url[[, row], callback]);
//which is equivalent to:
d3.request(url)
  .mimeType("text/csv")
  .response(function(xhr) { return d3.csvParse(xhr.responseText, row); })
  .get(callback);
  
function row(d) {
  return {
  //data manipulation here...
  };
}  

////Read tsv file
//Direct call:
d3.tsv(url[[, row], callback]);
//which is equivalent to:
d3.request(url)
  .mimeType("text/tab-separated-values")
  .response(function(xhr) { return d3.tsvParse(xhr.responseText, row); })
  .get(callback);

function row(d) {
  return {
  //data manipulation here...
  };
}  

////Read other delimited-separated file
var dlm = d3.dsvFormat("delimited"); //create formatter with specific delimiter
d3.request(url)
  .mimeType("text/plain")
  .response(function(xhr) { return d3.dlmParse(xhr.responseText, row); })
  .get(callback);

function row(d) {
  return {
  //data manipulation here...
  };
}  

////Read text file  
//Direct call:
d3.text(url[, callback]);
//which is equivalent to:
d3.request(url)
  .mimeType("text/plain")
  .response(function(xhr) { return xhr.responseText; })
  .get(callback);
    
////Read json file  
//Direct call:
d3.json(url[, callback]);
//which is equivalent to:
d3.request(url)
  .mimeType("application/json")
  .response(function(xhr) { return JSON.parse(xhr.responseText); })
  .get(callback);
    
////Read html file
//Direct call:
d3.html(url[, callback]);
//which is equivalent to:
d3.request(url)
  .mimeType("text/html")
  .response(function(xhr) { return document.createRange().createContextualFragment(xhr.responseText); })
  .get(callback);  
  
////Read xml file
//Direct call:
d3.xml(url[, callback]);
//which is equivalent to:
d3.request(url)
  .mimeType("application/xml")
  .response(function(xhr) { return xhr.responseXML; })
  .get(callback);
```  
<br />

<u>Reference:</u>    
----------------- 
(1). D3 API Reference, _[https://github.com/d3/d3/blob/master/API.md](https://github.com/d3/d3/blob/master/API.md)_.  
(2). d3-request API, _[https://github.com/d3/d3-request](https://github.com/d3/d3-request)_.   
(3). Data Loading in D3, _[http://www.tutorialsteacher.com/d3js/loading-data-from-file-in-d3js](http://www.tutorialsteacher.com/d3js/loading-data-from-file-in-d3js)_.  
(4). Learn JS Data, _[http://learnjsdata.com/read_data.html](http://learnjsdata.com/read_data.html)_.

[1]: https://d3js.org/  "d3"