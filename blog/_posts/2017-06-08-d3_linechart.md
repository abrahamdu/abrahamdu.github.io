---
layout: post
section-type: post
published: true
category: VISUALIZATION
tags: [ 'd3', 'html', 'css', 'javascript' ]
title: 'D3 Line Chart'
date: 2017-06-08
modifiedOn: 2017-07-21
---

Line chart is also quite commom in reality for people to track changes, especially continuous changes over short or long periods of time. [D3] [1] implements this by _d3.line()_ function under __[d3-shape](https://github.com/d3/d3-shape/blob/master/README.md#lines)__ module starting from version 4.0.  

__1. Simple Line Chart__  
------------------------  

The first simple line chart example basically takes the code from the [previous blog](http://anotherpeak.org/blog/visualization/2017/05/21/d3_histgram.html) to set up basic framework and random number generating process as well to start with.  

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#simple_line {
   background-color: lightgrey;
}

.line{
    fill: none;
    stroke: steelblue;
    stroke-width: 3;
}

.dot {
    fill: red;
}
</style>

<svg id="simple_line" width="950" height="500"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script>
var margin = {top: 30, right: 30, bottom: 30, left: 30};
var width = document.getElementById("simple_line").getBoundingClientRect().width-100;
var height = 400;

var data_x = d3.range(0,100,1),
    data_y = d3.range(0,100,1)
           .map(function(d) { return {"y": d3.randomUniform(0,10)() } });

var g = d3.select("#simple_line")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var xRange = d3.scaleLinear()
        .domain([d3.min(data_x),d3.max(data_x)])
        .rangeRound([0, width]);

var yRange = d3.scaleLinear()
    .domain([0,10])
    .range([height, 0]);

var line = d3.line()
    .x(function(d,i) { return xRange(i);  })
    .y(function(d) { return yRange(d.y);  })
    .curve(d3.curveMonotoneX);

g.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(xRange));

g.append("g")
      .attr("class", "y axis")
      .call(d3.axisLeft(yRange));

g.append("path")
      .data([data_y])
      .attr("class", "line") 
      .attr("d", line); 

g.selectAll(".dot")
    .data(data_y)
    .enter().append("circle") 
    .attr("class", "dot")
    .attr("cx", function(d, i) { return xRange(i); })
    .attr("cy", function(d) { return yRange(d.y); })
    .attr("r", 5);
</script>  

I changed the distribution to generate random numbers to uniform to avoid the headaches of unbounded limit from normal distribution. The key difference of drawing a line chart from histogram under [D3] [1] starts from the following code:  

```javascript   
var line = d3.line()
    .x(function(d,i) { return xRange(i);  })
    .y(function(d) { return yRange(d.y);  })
    .curve(d3.curveMonotoneX);
```  

_d3.line()_ is the line generator which uses an array of co-ordinates as inputs and ouputs a path data string. In this particular example, we also use _.x()_ and _.y()_ accessor functions by which we can specify how the line generator interprets each element of array. In this particular example, we use _.x()_ to define the index of the random numbers generated and _.y()_ to display the actual random numbers. _.curve()_ is to interpolate each data point. You can find the full set of curve factory here: [https://github.com/d3/d3-shape/blob/master/README.md#curves](https://github.com/d3/d3-shape/blob/master/README.md#curves).  

Here is the full code:  

```javascript  
<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#simple_line {
   background-color: lightgrey;
}

.line{
    fill: none;
    stroke: steelblue;
    stroke-width: 3;
}

.dot {
    fill: red;
}
</style>

<svg id="simple_line" width="950" height="500"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script>
var margin = {top: 30, right: 30, bottom: 30, left: 30};
var width = document.getElementById("simple_line").getBoundingClientRect().width-100;
var height = 400;

var data_x = d3.range(0,100,1),
    data_y = d3.range(0,100,1)
           .map(function(d) { return {"y": d3.randomUniform(0,10)() } });

var g = d3.select("#simple_line")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var xRange = d3.scaleLinear()
        .domain([d3.min(data_x),d3.max(data_x)])
        .rangeRound([0, width]);

var yRange = d3.scaleLinear()
    .domain([0,10])
    .range([height, 0]);

var line = d3.line()
    .x(function(d,i) { return xRange(i);  })
    .y(function(d) { return yRange(d.y);  })
    .curve(d3.curveMonotoneX);

g.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(xRange));

g.append("g")
      .attr("class", "y axis")
      .call(d3.axisLeft(yRange));

g.append("path")
      .data([data_y])
      .attr("class", "line") 
      .attr("d", line); 

g.selectAll(".dot")
    .data(data_y)
    .enter().append("circle") 
    .attr("class", "dot")
    .attr("cx", function(d, i) { return xRange(i); })
    .attr("cy", function(d) { return yRange(d.y); })
    .attr("r", 5);
</script>  
```  

__2. Real Line Chart Example - Browser Statistics from W3Schools__  
------------------------------------------------------------------  

As mentioned before, line chart is good to illustrate the trend over time and we will use the real data from [W3School Browser Statistics](https://www.w3schools.com/browsers/default.asp), as of June 2017, to show the popularity of web browsers over time. I choose not to use the data prior 2008 due to the frequency of data available before that time is not monthly.  

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#browser {
   background-color: lightgrey;
}

.line:hover{
    fill: none;
    stroke-width: 5;
}

.axis--x path {
  display: none;
}
</style>

<svg id="browser" width="950" height="500"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script>
var margin = {top: 30, right: 30, bottom: 30, left: 30};
var width = document.getElementById("browser").getBoundingClientRect().width-100;
var height = 400;

var parseTime = d3.timeParse("%b-%y");

var color = d3.scaleOrdinal(d3.schemeCategory10);

var g = d3.select("#browser")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var xRange = d3.scaleTime()
        .rangeRound([0, width]);

var yRange = d3.scaleLinear()
    .range([height, 0]);

var line = d3.line()
    .x(function(d) { return xRange(d.Date);  })
    .y(function(d) { return yRange(d.market_share);  })
    .curve(d3.curveMonotoneX);

d3.csv("/blog/data/browser_statistics.csv",function(d, i, columns) {
    d.Date = parseTime(d.Date);
    for (var i = 1, n = columns.length, c; i < n; ++i) d[c = columns[i]] = +d[c];
    return d;
    },
    function(error,data){
    if (error) throw error;
    var browsers = data.columns.slice(1).map(function(id) {
    return {
      id: id,
      values: data.map(function(d) {
      return {Date: d.Date, market_share: d[id]};
      })
    };     
  });

xRange.domain(d3.extent(data, function(d) { return d.Date; }));

yRange.domain([
    d3.min(browsers, function(c) { return d3.min(c.values, function(d) { return d.market_share; }); }),
    d3.max(browsers, function(c) { return d3.max(c.values, function(d) { return d.market_share; }); })
  ]);

color.domain(browsers.map(function(c) { return c.id; }));

 g.append("g")
      .attr("class", "axis")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(xRange))
      .style("text-anchor", "middle")
      .attr("dx", "-.1em");

  g.append("g")
      .attr("class", "axis")
      .call(d3.axisLeft(yRange))
      .append("text")
      .attr("x",2)
      .attr("y", 6)
      .attr("dy", "0.71em")
      .attr("fill", "#000")
      .attr("text-anchor", "start")
      .text("Market Share, %");

var browser = g.selectAll(".browser")
    .data(browsers)
    .enter()
    .append("g")
    .attr("class", "browser");

  browser.append("path")
      .attr("class", "line")
      .attr("d", function(d) { return line(d.values); })
      .style("stroke", function(d) { return color(d.id); });
      
  browser.append("text")
      .datum(function(d) { return {id: d.id, value: d.values[0]}; })
      .attr("transform", function(d) { return "translate(" + xRange(d.value.Date) + "," + yRange(d.value.market_share) + ")"; })
      .attr("x", 3)
      .attr("dy", "0.35em")
      .style("font", "10px sans-serif")
      .text(function(d) { return d.id; });

});

</script>  

The code is pretty much alike the one used to draw histogram  except the call to prepare the line function.  

```javascript  
var line = d3.line()
    .x(function(d) { return xRange(d.Date);  })
    .y(function(d) { return yRange(d.market_share);  })
    .curve(d3.curveMonotoneX);  
```  

And append the path by using the line function created above.  

```javascript  
browser.append("path")
      .attr("class", "line")
      .attr("d", function(d) { return line(d.values); })
      .style("stroke", function(d) { return color(d.id); });
```  
Here is the full code:  

```javascript  
<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#browser {
   background-color: lightgrey;
}

.line:hover{
    fill: none;
    stroke-width: 5;
}

.axis--x path {
  display: none;
}
</style>

<svg id="browser" width="950" height="500"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script>
var margin = {top: 30, right: 30, bottom: 30, left: 30};
var width = document.getElementById("browser").getBoundingClientRect().width-100;
var height = 400;

var parseTime = d3.timeParse("%b-%y");

var color = d3.scaleOrdinal(d3.schemeCategory10);

var g = d3.select("#browser")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var xRange = d3.scaleTime()
        .rangeRound([0, width]);

var yRange = d3.scaleLinear()
    .range([height, 0]);

var line = d3.line()
    .x(function(d) { return xRange(d.Date);  })
    .y(function(d) { return yRange(d.market_share);  })
    .curve(d3.curveMonotoneX);

d3.csv("/blog/data/browser_statistics.csv",function(d, i, columns) {
    d.Date = parseTime(d.Date);
    for (var i = 1, n = columns.length, c; i < n; ++i) d[c = columns[i]] = +d[c];
    return d;
    },
    function(error,data){
    if (error) throw error;
    var browsers = data.columns.slice(1).map(function(id) {
    return {
      id: id,
      values: data.map(function(d) {
      return {Date: d.Date, market_share: d[id]};
      })
    };     
  });

xRange.domain(d3.extent(data, function(d) { return d.Date; }));

yRange.domain([
    d3.min(browsers, function(c) { return d3.min(c.values, function(d) { return d.market_share; }); }),
    d3.max(browsers, function(c) { return d3.max(c.values, function(d) { return d.market_share; }); })
  ]);

color.domain(browsers.map(function(c) { return c.id; }));

 g.append("g")
      .attr("class", "axis")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(xRange))
      .style("text-anchor", "middle")
      .attr("dx", "-.1em");

  g.append("g")
      .attr("class", "axis")
      .call(d3.axisLeft(yRange))
      .append("text")
      .attr("x",2)
      .attr("y", 6)
      .attr("dy", "0.71em")
      .attr("fill", "#000")
      .attr("text-anchor", "start")
      .text("Market Share, %");

var browser = g.selectAll(".browser")
    .data(browsers)
    .enter()
    .append("g")
    .attr("class", "browser");

  browser.append("path")
      .attr("class", "line")
      .attr("d", function(d) { return line(d.values); })
      .style("stroke", function(d) { return color(d.id); });
      
  browser.append("text")
      .datum(function(d) { return {id: d.id, value: d.values[0]}; })
      .attr("transform", function(d) { return "translate(" + xRange(d.value.Date) + "," + yRange(d.value.market_share) + ")"; })
      .attr("x", 3)
      .attr("dy", "0.35em")
      .style("font", "10px sans-serif")
      .text(function(d) { return d.id; });

});

</script>  
```  

<u>Reference:</u>    
----------------- 
(1). D3 API Reference, _[https://github.com/d3/d3/blob/master/API.md](https://github.com/d3/d3/blob/master/API.md)_.   
(2). Multi-Series Line Chart, _[https://bl.ocks.org/mbostock/3884955](https://bl.ocks.org/mbostock/3884955)_.  

[1]: https://d3js.org/  "d3"