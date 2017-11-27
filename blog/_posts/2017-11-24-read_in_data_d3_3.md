---
layout: post
section-type: post
published: true
category: VISUALIZATION
tags: [ 'd3', 'javascript' ]
title: 'Read in Data by D3 - Part 3: D3 Queue'
date: 2017-11-24
modifiedOn: 2017-11-26
---  

It is more common that people would use a quite few number of files at the same session for data visualization for various reasons and it is critical to ascertain that [D3] [1] runs after all files are finished loading and ready for use. __d3-queue__ comes to play a role in loading external files into the script and ensuring each file is completely loaded before allowing the script to continue running because of _asynchronous_. The whole suite of __d3-queue__ would let user to load multiple files and return callbacks once completion with the order it was requested.    

I am going to use the files used in the previous blogs for illustration.  

```javascript 
<script src="https://d3js.org/d3-collection.v1.min.js"></script>
<script src="https://d3js.org/d3-dispatch.v1.min.js"></script>
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script src="https://d3js.org/d3-request.v1.min.js"></script>
<script src="https://d3js.org/d3-queue.v3.min.js"></script>
<script>
  d3.queue()
    .defer(d3.csv,"/blog/data/browser_statistics.csv")
    .defer(d3.text,"/blog/data/age_by_gender.txt")
    .defer(d3.json,"/blog/data/all_countries_2015.json")
    .defer(d3.xml,"/blog/data/example.xml")
    .awaitAll(analyze);

function analyze(error, data) {
  if(error) throw error;
  console.log(data);
}
//(4) [Array(118), "Age_Group;Male;Female↵Under 5 years;0.064;0.059↵…ears;0.015;0.021↵85 years andover;0.013;0.025↵", {…}, document]
//0: (118) [{…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, …]
//1: "Age_Group;Male;Female↵Under 5 years;0.064;0.059↵5 to 9 years;0.066;0.062↵10 to 14 years;0.067;0.062 ..."
//2: {data: {…}, children: Array(6)}
//3: document
//length: 4
//__proto__: Array(0)
</script>
```  
Similarly, it starts with _d3.queue()_ function to construct a new queue with no argument. By default, it would have value 1 which run all the tasks followed by in series. If you want to specify number of concurrency of how many tasks could be run concurrently, you can put any positive number in it. _defer()_ function adds the specified asynchronous task callback to the queue. In the example above, it uses different functions from __d3-request__ module to pass multiple types of flat files into the queue in order and sets the callback to be invoked when all the deferred tasks have finished by _awaitAll()_ function. The returned data is an array of four elements with the same order in the queue. There is also a variant of _awaitAll()_ function, which is _await()_. Instead of passing all the results into one callback, it would pass each result from loading each file into individual callback, as shown below:    

```javascript  
<script src="https://d3js.org/d3-collection.v1.min.js"></script>
<script src="https://d3js.org/d3-dispatch.v1.min.js"></script>
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script src="https://d3js.org/d3-request.v1.min.js"></script>
<script src="https://d3js.org/d3-queue.v3.min.js"></script>
<script>
  d3.queue()
    .defer(d3.csv,"/blog/data/browser_statistics.csv")
    .defer(d3.text,"/blog/data/age_by_gender.txt")
    .defer(d3.json,"/blog/data/all_countries_2015.json")
    .defer(d3.xml,"/blog/data/example.xml")
    .await(analyze);

function analyze(error, csv, txt, json, xml) {
  if(error) throw error;
  console.log(csv); //(118) [{…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, {…}, …]
  console.log(txt); //Age_Group;Male;Female ...
  console.log(json); //{data: {…}, children: Array(6)}
  console.log(xml);  //#document
}
</script>  
```  
Using __d3-queue__ is actually quite easy and if there is only one file used in the [D3] [1] script, there is no need to use it. If it requires many callbacks, we should consider using it. And if we want to control number of concurrent tasks of callbacks to make sure a request won't overload the server, the flexibility is given by the library to set the number of concurrency in the _d3.queue()_ function. It would make sure loading multiple files would be completed once it is called with the order we put into the queue.  

<br />

<u>Reference:</u>    
----------------- 
(1). D3 API Reference, _[https://github.com/d3/d3/blob/master/API.md](https://github.com/d3/d3/blob/master/API.md)_.  
(2). d3-queue API, _[https://github.com/d3/d3-queue](https://github.com/d3/d3-queue)_.   
(3). Learn JS Data, _[http://learnjsdata.com/read_data.html](http://learnjsdata.com/read_data.html)_.  
(4). A mental model of what d3-queue does, _[https://macwright.org/2016/12/09/a-d3-queue-mental-model.html](https://macwright.org/2016/12/09/a-d3-queue-mental-model.html)_.

[1]: https://d3js.org/  "d3"