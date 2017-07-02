---
layout: post
section-type: post
published: true
category: VISUALIZATION
tags: [ 'd3', 'html', 'css', 'javascript' ]
title: 'D3 Area Chart'
date: 2017-06-28
modifiedOn: 2017-06-28
---

Quite similar to line chart, area chart could also be used to demonstrate the trends over time for comparison. In this post, I'd like to use the same data in [last blog](http://anotherpeak.org/blog/visualization/2017/06/08/d3_linechart.html) from [W3School Browser Statistics](https://www.w3schools.com/browsers/default.asp), but to draw an area chart to show the market share change by different browsers.  

As you would see, area chart by [D3] [1] uses _d3.area()_ function, also under __[d3-shape](https://github.com/d3/d3-shape/blob/master/README.md#areas)__ module.  

```javascript  
var area = d3.area()
    .x(function(d,i) { return xRange(d.data.Date);  })
    .y0(function(d) { return yRange(d[0]);  })
    .y1(function(d) { return yRange(d[1]);  });  
```  
Essentially, I use _d3.area()_ to generate an area which is defined by two bounding lines. In this specific example, the two lines share the same x-value which is time series value. However, the y-values differ by _y0_ and _y1_. It uses curveLinear curve to produce a clockwise polygon and (_x1_, _y1_) renders the topline first then (_x0_, _y0_) render the bottomline later.  

This _area_ function is called later to generate the paths.  

```javascript  
browser.append("path")
      .attr("class", "area")
      .style("fill", function(d) { return color(d.key); })
      .attr("d", area);
```  

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#browser {
   background-color: lightgrey;
}

.browser:hover{
    fill: darkgreen;
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

var stack = d3.stack();

var area = d3.area()
    .x(function(d,i) { return xRange(d.data.Date);  })
    .y0(function(d) { return yRange(d[0]);  })
    .y1(function(d) { return yRange(d[1]);  });

d3.csv("/blog/data/browser_statistics.csv",function(d, i, columns) {
    d.Date = parseTime(d.Date);
    for (var i = 1, n = columns.length, c; i < n; ++i) d[c = columns[i]] = +d[c];
    return d;
    },
    function(error,data){
    if (error) throw error;
    
var browsers = data.columns.slice(1);

xRange.domain(d3.extent(data, function(d) { return d.Date; }));
yRange.domain([0,100]);
color.domain(browsers);

stack.keys(browsers);

var browser = g.selectAll(".browser")
    .data(stack(data))
    .enter()
    .append("g")
    .attr("class", "browser");

  browser.append("path")
      .attr("class", "area")
      .style("fill", function(d) { return color(d.key); })
      .attr("d", area);

  browser.append('text')
      .datum(function(d) { return d; })
      .attr('transform', function(d) { return 'translate(' + xRange(data[0].Date) + ',' + yRange(d[0][0]) + ')'; })
      .attr('x', 3) 
      .attr('dy', '.35em')
      .style("font", "10px sans-serif")
      .text(function(d) { return d.key; });

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
      .attr("x",3)
      .attr("y", 6)
      .attr("dy", "0.71em")
      .attr("fill", "#000")
      .attr("text-anchor", "start")
      .text("Market Share, %")
      .attr('fill-opacity', 1);
});

</script>  

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

.browser:hover{
    fill: darkgreen;
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

var stack = d3.stack();

var area = d3.area()
    .x(function(d,i) { return xRange(d.data.Date);  })
    .y0(function(d) { return yRange(d[0]);  })
    .y1(function(d) { return yRange(d[1]);  });

d3.csv("/blog/data/browser_statistics.csv",function(d, i, columns) {
    d.Date = parseTime(d.Date);
    for (var i = 1, n = columns.length, c; i < n; ++i) d[c = columns[i]] = +d[c];
    return d;
    },
    function(error,data){
    if (error) throw error;
    
var browsers = data.columns.slice(1);

xRange.domain(d3.extent(data, function(d) { return d.Date; }));
yRange.domain([0,100]);
color.domain(browsers);

stack.keys(browsers);

var browser = g.selectAll(".browser")
    .data(stack(data))
    .enter()
    .append("g")
    .attr("class", "browser");

  browser.append("path")
      .attr("class", "area")
      .style("fill", function(d) { return color(d.key); })
      .attr("d", area);

  browser.append('text')
      .datum(function(d) { return d; })
      .attr('transform', function(d) { return 'translate(' + xRange(data[0].Date) + ',' + yRange(d[0][0]) + ')'; })
      .attr('x', 3) 
      .attr('dy', '.35em')
      .style("font", "10px sans-serif")
      .text(function(d) { return d.key; });

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
      .attr("x",3)
      .attr("y", 6)
      .attr("dy", "0.71em")
      .attr("fill", "#000")
      .attr("text-anchor", "start")
      .text("Market Share, %")
      .attr('fill-opacity', 1);
});

</script>  
```  

</script>  
<u>Reference:</u>    
----------------- 
(1). D3 API Reference, _[https://github.com/d3/d3/blob/master/API.md](https://github.com/d3/d3/blob/master/API.md)_.   
(2). Multi-Series Line Chart, _[https://bl.ocks.org/mbostock/3884955](https://bl.ocks.org/mbostock/3884955)_.  

[1]: https://d3js.org/  "d3"