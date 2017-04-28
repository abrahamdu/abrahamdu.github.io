---
layout: post
section-type: post
published: true
category: VISUALIZATION
tags: [ 'd3', 'html', 'css', 'javascript' ]
title: 'D3 Examples and Building Blocks'
date: 2016-09-18
modifiedOn: 2017-04-20
---

Before we dig into each [D3] [1] modules, let's first take a few simple [D3] [1] examples to get a main idea of the code. The followings are some examples I pull from [Bl.ock Builder](http://blockbuilder.org/).  

Example 1: [bar chart](http://bl.ocks.org/mbostock/7322386)  

```javascript   
<!DOCTYPE html>
<meta charset="utf-8">
<style>

.chart div {
  font: 10px sans-serif;
  background-color: steelblue;
  text-align: right;
  padding: 3px;
  margin: 1px;
  color: white;
}

</style>
<div class="chart"></div>
<script src="//d3js.org/d3.v3.min.js"></script>
<script>

var data = [4, 8, 15, 16, 23, 42];

var x = d3.scale.linear()
    .domain([0, d3.max(data)])
    .range([0, 420]);

d3.select(".chart")
  .selectAll("div")
    .data(data)
  .enter().append("div")
    .style("width", function(d) { return x(d) + "px"; })
    .text(function(d) { return d; });

</script>
```   
Example 2: [line chart](https://bl.ocks.org/mbostock/3883245)  

```javascript  
<!DOCTYPE html>
<meta charset="utf-8">
<style>

body {
  font: 10px sans-serif;
}

.axis path,
.axis line {
  fill: none;
  stroke: #000;
  shape-rendering: crispEdges;
}

.x.axis path {
  display: none;
}

.line {
  fill: none;
  stroke: steelblue;
  stroke-width: 1.5px;
}

</style>
<body>
<script src="//d3js.org/d3.v3.min.js"></script>
<script>

var margin = {top: 20, right: 20, bottom: 30, left: 50},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

var formatDate = d3.time.format("%d-%b-%y");

var x = d3.time.scale()
    .range([0, width]);

var y = d3.scale.linear()
    .range([height, 0]);

var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom");

var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left");

var line = d3.svg.line()
    .x(function(d) { return x(d.date); })
    .y(function(d) { return y(d.close); });

var svg = d3.select("body").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

d3.tsv("data.tsv", type, function(error, data) {
  if (error) throw error;

  x.domain(d3.extent(data, function(d) { return d.date; }));
  y.domain(d3.extent(data, function(d) { return d.close; }));

  svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis);

  svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("Price ($)");

  svg.append("path")
      .datum(data)
      .attr("class", "line")
      .attr("d", line);
});

function type(d) {
  d.date = formatDate.parse(d.date);
  d.close = +d.close;
  return d;
}

</script>
```

Example 3: [world map](http://bl.ocks.org/d3noob/5189184)  

```javascript
<!DOCTYPE html>
<meta charset="utf-8">
<style>
path {
  stroke: white;
  stroke-width: 0.25px;
  fill: grey;
}
</style>
<body>
<script src="http://d3js.org/d3.v3.min.js"></script>
<script src="http://d3js.org/topojson.v0.min.js"></script>
<script>
var width = 960,
    height = 500;

var projection = d3.geo.mercator()
    .center([0, 5 ])
    .scale(150)
    .rotate([-180,0]);

var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height);

var path = d3.geo.path()
    .projection(projection);

var g = svg.append("g");

// load and display the World
d3.json("world-110m2.json", function(error, topology) {
    g.selectAll("path")
      .data(topojson.object(topology, topology.objects.countries)
          .geometries)
    .enter()
      .append("path")
      .attr("d", path)
});

</script>
</body>
</html>
```

From the three [D3] [1] examples above we can see, three are three major components of [D3] [1] project: HTML, CSS and JavaScript.  

__1. HTML__ 
-----------  

```html
<!DOCTYPE html>
<meta charset="utf-8">

<body>

</body>
</html>
```

The Hypertext Markup Language (HTML) resource is the main markup language for displaying web pages. HTML elements are the building blocks of the HTML web page. The elements consist of a pair of tags (starting and ending tags) and the textual or graphical content inside of the tags. For example, *<body>* tag defines the document's body.  

__2. CSS__  
----------   

```css  
/*Example 1*/
<style>

.chart div {
  font: 10px sans-serif;
  background-color: steelblue;
  text-align: right;
  padding: 3px;
  margin: 1px;
  color: white;
}

</style>

/*Example 2*/
<style>

body {
  font: 10px sans-serif;
}

.axis path,
.axis line {
  fill: none;
  stroke: #000;
  shape-rendering: crispEdges;
}

.x.axis path {
  display: none;
}

.line {
  fill: none;
  stroke: steelblue;
  stroke-width: 1.5px;
}

</style>

/*Example 3*/
<style>

path {
  stroke: white;
  stroke-width: 0.25px;
  fill: grey;
}

</style>
```    

The Cascading Style Sheet (CSS) resource is the style sheet language used for describing the presentation of the document. The presentation includes both the look as well as the formatting. There are three ways of adding CSS formatting to HTML elements. The examples above show a way, putting CSS within the <em>\<style\></em> element which is called as 'Internal Style Sheet'.   

You can also separate all our CSS codes into a standalone piece and let HTML to call it by using the <em>\<link\></em> element. For example:  

```css
<link rel="stylesheet" type="text/css" href="mystyle.css">
```

This is so-called 'External Style Sheet'.  

The third way of add CSS formatting is 'Inline Styles' which play as an optional parameter of HTML tags by only changing its styles. For example:  

```css
<h1 style="color:blue;margin-left:30px;">This is a heading.</h1>
```   

__3. JavaScript__  
-----------------  

If we say HTML defines the contents of web page and CSS formats it, JavaScript is the programming language which gives web page functionality.  In the examples, the remaining codes not belonging to HTML or CSS are basically JavaScript codes with some [D3] [1] flavors. Like CSS, there are also a couple of places where JavaScript could be embedded.  

You can call in JavaScript from external sources and include it inside <em>\<script\></em> tag under *<body>* tag, like Example 3:  

```javascript 
<body>
<script src="http://d3js.org/d3.v3.min.js"></script>
<script src="http://d3js.org/topojson.v0.min.js"></script>
</body>
```  

Or instead of calling external JavaScript code, you can place JavaScript code directly inside <em>\<script\></em> tag. In the three examples above, the longest codes inside <em>\<script\></em> tags are used by this way.  

A third way, which is not shown in our examples, is to put JavaScript inside *<head>* tag which is outside the scope of *<head>* tag.   

All of three, HTML, CSS and JavaScript, are different programming languages themselves. If you feel interested in study more, please check reference 1 - 3 for details.  

__4. DOM, SVG and Canvas__  
--------------------------  

Document Object Model (DOM) is a convention for representing and interacting with objects in HTML, XML, and XHTML documents. The DOM is separated into three parts: Core, HTML, and XML. The DOM allows programs and scripts to dynamically access and update the structure, content, and style of a document.  

Scalable Vector Graphics (SVG) is a family of specifications for creating two-dimensional vector graphics. Vector graphics are not created out of pixels. Vector graphics are created with paths having a start and end point, as well as points, curves and angles in between. Since Vector Graphics are not created out of pixels, they can be scaled up to larger or smaller sizes without losing image quality. Tag <em>\<svg\></em> is a container for SVG graphics.    

Canvas is also a language from drawing 2D graphics, but pixel by pixel. Different from SVG, Canvas is essentially a graph container when moving between different browsers, you must call a script (usually JavaScript) to draw graphics. Tag <em>\<canvas\></em> is a container for Canvas graphics.  

__5. Browser Support__  
----------------------  

Finally, we need a browser to present our [D3] [1] project. According to [D3] [1] [wiki](https://github.com/d3/d3/wiki), majority 'modern' browsers support [D3] [1] except IE8 and older versions.  


Let's start our [D3] [1] journey!

<br />

<u>Reference:</u>    
-----------------  
(1). HTML5 Tutorial from W3School.com, _[http://www.w3schools.com/html/default.asp](http://www.w3schools.com/html/default.asp)_.  
(2). CSS Tutorial from W3School.com, _[http://www.w3schools.com/css/default.asp](http://www.w3schools.com/css/default.asp)_.  
(3). JavaScript Tutorial from W3School.com, _[http://www.w3schools.com/js/default.asp](http://www.w3schools.com/js/default.asp)_.    

[1]: https://d3js.org/  "d3"