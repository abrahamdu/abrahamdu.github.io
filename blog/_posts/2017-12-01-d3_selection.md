---
layout: post
section-type: post
published: true
category: VISUALIZATION
tags: [ 'd3', 'javascript' ]
title: 'D3 Selections: Enter, Update and Exit'
date: 2017-12-01
modifiedOn: 2017-12-01
---  

[D3] [1]'s selection plays a core function of its full suite. We all know [D3] [1] stands for data-driven documents which means the visualization is based on data to manipulate [DOM](https://en.wikipedia.org/wiki/Document_Object_Model). __d3-selection__ module would manipulate _DOM_ to bind data with it for changing style, modifying attributes or updating/inserting/removing elements. Let's first generate a series of shapes to start with. For simplicity, I use _[Symbols](https://github.com/d3/d3-shape#symbols)_ under __d3-shape__ module to generate the following seven symbols.  

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#example {
   background-color: lightgrey;
}

</style>

<svg id="example" width="950" height="150"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script>
var margin = {top: 60, right: 50, bottom: 30, left: 50};
var width = document.getElementById("example").getBoundingClientRect().width-100;
var height = 150;
    
var g1 = d3.select("#example")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");    
g1.append('path')
  .attr('d', d3.symbol().type(d3.symbolCircle).size(2000))
  .attr("fill", "black");

var g2 = d3.select("#example")
        .append("g")
        .attr("transform", "translate(" + margin.left*3 +"," + margin.top + ")"); 
g2.append('path')
  .attr('d', d3.symbol().type(d3.symbolCross).size(2000))
  .attr("fill", "black");

var g3 = d3.select("#example")
        .append("g")
        .attr("transform", "translate(" + margin.left*6 +"," + margin.top + ")"); 
g3.append('path')
  .attr('d', d3.symbol().type(d3.symbolDiamond).size(2000))
  .attr("fill", "black");
 
var g4 = d3.select("#example")
        .append("g")
        .attr("transform", "translate(" + margin.left*9 +"," + margin.top + ")"); 
g4.append('path')
  .attr('d', d3.symbol().type(d3.symbolSquare).size(2000))
  .attr("fill", "black");

var g5 = d3.select("#example")
        .append("g")
        .attr("transform", "translate(" + margin.left*12 +"," + margin.top + ")"); 
g5.append('path')
  .attr('d', d3.symbol().type(d3.symbolStar).size(2000))
  .attr("fill", "black");
 
var g6 = d3.select("#example")
        .append("g")
        .attr("transform", "translate(" + margin.left*15 +"," + margin.top + ")"); 
g6.append('path')
  .attr('d', d3.symbol().type(d3.symbolTriangle).size(2000))
  .attr("fill", "black");
 
var g7 = d3.select("#example")
        .append("g")
        .attr("transform", "translate(" + margin.left*18 +"," + margin.top + ")"); 
g7.append('path')
  .attr('d', d3.symbol().type(d3.symbolWye).size(2000))
  .attr("fill", "black");
</script> 

If you open the developer console of browser, you will find the _DOM_ structure as follows:  

![alt text](/img/blog/d3_selection1.png) 

__1. Selection__ 
----------------  

Selections allow us to select _DOM_ elements with compliance to _CSS_ selectors. For example, _d3.select()_ would select the first matching element in the specified string.  

```javascript  
selectionDiv = d3.select("#example"); // select the first DOM element with #example
selectionG = d3.select("#example).select("g"); // select the fisrt g child from #example
selectionAllG = d3.select("#example).selectAll("g"); // select all g children from #example
```  
![alt text](/img/blog/d3_selection2.png)

Once we have a selection of elements, we can apply operators to it in order to manipulate things such as styles and attributes.  

```javascript  
d3.select("#example").select("g").select("path").attr("fill","yellow"); // change the first g element into yellow
```  
<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#example1 {
   background-color: lightgrey;
}

</style>

<svg id="example1" width="950" height="150"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script>
var margin = {top: 60, right: 50, bottom: 30, left: 50};
var width = document.getElementById("example1").getBoundingClientRect().width-100;
var height = 150;
    
var g1 = d3.select("#example1")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");    
g1.append('path')
  .attr('d', d3.symbol().type(d3.symbolCircle).size(2000))
  .attr("fill", "black");

var g2 = d3.select("#example1")
        .append("g")
        .attr("transform", "translate(" + margin.left*3 +"," + margin.top + ")"); 
g2.append('path')
  .attr('d', d3.symbol().type(d3.symbolCross).size(2000))
  .attr("fill", "black");

var g3 = d3.select("#example1")
        .append("g")
        .attr("transform", "translate(" + margin.left*6 +"," + margin.top + ")"); 
g3.append('path')
  .attr('d', d3.symbol().type(d3.symbolDiamond).size(2000))
  .attr("fill", "black");
 
var g4 = d3.select("#example1")
        .append("g")
        .attr("transform", "translate(" + margin.left*9 +"," + margin.top + ")"); 
g4.append('path')
  .attr('d', d3.symbol().type(d3.symbolSquare).size(2000))
  .attr("fill", "black");

var g5 = d3.select("#example1")
        .append("g")
        .attr("transform", "translate(" + margin.left*12 +"," + margin.top + ")"); 
g5.append('path')
  .attr('d', d3.symbol().type(d3.symbolStar).size(2000))
  .attr("fill", "black");
 
var g6 = d3.select("#example1")
        .append("g")
        .attr("transform", "translate(" + margin.left*15 +"," + margin.top + ")"); 
g6.append('path')
  .attr('d', d3.symbol().type(d3.symbolTriangle).size(2000))
  .attr("fill", "black");
 
var g7 = d3.select("#example1")
        .append("g")
        .attr("transform", "translate(" + margin.left*18 +"," + margin.top + ")"); 
g7.append('path')
  .attr('d', d3.symbol().type(d3.symbolWye).size(2000))
  .attr("fill", "black");
  
d3.select("#example1").select("g").select("path").attr("fill","yellow");
</script>  

Selections would also allow us to create new _DOM_ elements. For example, imagine we have a series of data, __[1,2,3,4,5,6,7]__ that we want to assign to each symbol as labels. We can rely on the _selection_ introduced above to first select the _DOM_ element from which we want to add the texts.  

```javascript  
var label = [1,2,3,4,5,6,7];
d3.select("#example")
  .selectAll("text")
  .data(label)
  .enter()
  .append("text")
  .attr("fill","red")
  .attr("font-size","30")
  .attr("x",function(d,i){
                          if(i==0) {return margin.left-10;}
                          else {return margin.left*3*(d-1)-10;}
                          })
  .attr("y",140)
  .text(function(d){return d;});
```  

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#example2 {
   background-color: lightgrey;
}

</style>

<svg id="example2" width="950" height="150"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script>
var margin = {top: 60, right: 50, bottom: 30, left: 50};
var width = document.getElementById("example2").getBoundingClientRect().width-100;
var height = 150;
    
var g1 = d3.select("#example2")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");    
g1.append('path')
  .attr('d', d3.symbol().type(d3.symbolCircle).size(2000))
  .attr("fill", "black");

var g2 = d3.select("#example2")
        .append("g")
        .attr("transform", "translate(" + margin.left*3 +"," + margin.top + ")"); 
g2.append('path')
  .attr('d', d3.symbol().type(d3.symbolCross).size(2000))
  .attr("fill", "black");

var g3 = d3.select("#example2")
        .append("g")
        .attr("transform", "translate(" + margin.left*6 +"," + margin.top + ")"); 
g3.append('path')
  .attr('d', d3.symbol().type(d3.symbolDiamond).size(2000))
  .attr("fill", "black");
 
var g4 = d3.select("#example2")
        .append("g")
        .attr("transform", "translate(" + margin.left*9 +"," + margin.top + ")"); 
g4.append('path')
  .attr('d', d3.symbol().type(d3.symbolSquare).size(2000))
  .attr("fill", "black");

var g5 = d3.select("#example2")
        .append("g")
        .attr("transform", "translate(" + margin.left*12 +"," + margin.top + ")"); 
g5.append('path')
  .attr('d', d3.symbol().type(d3.symbolStar).size(2000))
  .attr("fill", "black");
 
var g6 = d3.select("#example2")
        .append("g")
        .attr("transform", "translate(" + margin.left*15 +"," + margin.top + ")"); 
g6.append('path')
  .attr('d', d3.symbol().type(d3.symbolTriangle).size(2000))
  .attr("fill", "black");
 
var g7 = d3.select("#example2")
        .append("g")
        .attr("transform", "translate(" + margin.left*18 +"," + margin.top + ")"); 
g7.append('path')
  .attr('d', d3.symbol().type(d3.symbolWye).size(2000))
  .attr("fill", "black");
  
d3.select("#example2").select("g").select("path").attr("fill","yellow");

var label = [1,2,3,4,5,6,7];
d3.select("#example2")
  .selectAll("text")
  .data(label)
  .enter()
  .append("text")
  .attr("fill","red")
  .attr("font-size","30")
  .attr("x",function(d,i){
                          if(i==0) {return margin.left-10;}
                          else {return margin.left*3*(d-1)-10;}
                          })
  .attr("y",140)
  .text(function(d){return d;});
</script>  

When checking the browser console, we would find the first _g_ element with modified color "yellow" and there are seven _text_ elements created within _svg_ element _#example_.  

![alt text](/img/blog/d3_selection3.png) 

__2. Data Join__ 
---------------- 

If we re-visit the above example, you will find out the way how I made the seven symbols. Here is a snippet of my codes:  

```javascript   
var g1 = d3.select("#example2")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");    
g1.append('path')
  .attr('d', d3.symbol().type(d3.symbolCircle).size(2000))
  .attr("fill", "black");
.
.
.
var g7 = d3.select("#example2")
        .append("g")
        .attr("transform", "translate(" + margin.left*18 +"," + margin.top + ")"); 
g7.append('path')
  .attr('d', d3.symbol().type(d3.symbolWye).size(2000))
  .attr("fill", "black");
```  

Basically, I use _d3.select()_ to get _DOM_ element where we want to draw symbols and then _append_ each symbol to the _svg_ element by setting its position and specific type with seven times manually.  

Obviously, this is quite manual and inefficient way to visualize many graphs. And it didn't really need [D3] [1] to do it. Now, we want to make use of _selection_ introduced above the achieve the same goal. In order to be data-driven _DOM_, we also want to bind the data with it for visualization.    

```javascript  
var label = [4/3,2,3,4,5,6,7];
var color = d3.scaleOrdinal(d3.schemeCategory10);
    
var svg = d3.select("#example")
            .append("g")
            .attr("width",width)
            .attr("height",height);

svg.selectAll("path")
   .data(label)
   .enter()
   .append("path")
   .attr("transform", function(d,i) { return "translate("+margin.left*(d-1)*3+","+margin.top+")";})
   .attr('d', d3.symbol().type(function(d,i) { return d3.symbols[i];}).size(function(d,i) {return 500*d}))
   .attr("fill", d=>color(d));

var text = [1,2,3,4,5,6,7];
d3.select("#example")
  .selectAll("text")
  .data(text)
  .enter()
  .append("text")
  .attr("fill","red")
  .attr("font-size","30")
  .attr("x",function(d,i){
                          if(i==0) {return margin.left-10;}
                          else {return margin.left*3*(d-1)-10;}
                          })
  .attr("y",140)
  .text(function(d){return d;});
```  

First of all, we create two arrays: _label_ for creating number of symbols and _color_ for different colors using the data in _label_ array. _selectAll("path")_ would match existing paths or create new paths if not existing. _data(label)_ would join those paths with _label_. In other words, we bind each element from _label_ array to each _path_ element and by opening browser console, we get the following information for confirmation.  

![alt text](/img/blog/d3_selection4.png) 

It is noticed that from console, there are two attributes: __enter__ and __exit__. Here, we use _enter()_ and _append(path)_ to create the seven path elements by appending them for the new data. And the following three _attr()_ statements are just use data and index from the _data(label)_ to determine the position of each symbol, the size and color of it. Here is the final visualization:    

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#example3 {
   background-color: lightgrey;
}

</style>

<svg id="example3" width="950" height="150"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script>
var margin = {top: 60, right: 50, bottom: 30, left: 50};
var width = document.getElementById("example3").getBoundingClientRect().width-100;
var height = 150;
var label = [4/3,2,3,4,5,6,7];
var color = d3.scaleOrdinal(d3.schemeCategory10);
    
var svg = d3.select("#example3")
            .append("g")
            .attr("width",width)
            .attr("height",height);

svg.selectAll("path")
   .data(label)
   .enter()
   .append("path")
   .attr("transform", function(d,i) { return "translate("+margin.left*(d-1)*3+","+margin.top+")";})
   .attr('d', d3.symbol().type(function(d,i) { return d3.symbols[i];}).size(function(d,i) {return 500*d}))
   .attr("fill", d=>color(d));

var text = [1,2,3,4,5,6,7];
d3.select("#example3")
  .selectAll("text")
  .data(text)
  .enter()
  .append("text")
  .attr("fill","red")
  .attr("font-size","30")
  .attr("x",function(d,i){
                          if(i==0) {return margin.left-10;}
                          else {return margin.left*3*(d-1)-10;}
                          })
  .attr("y",140)
  .text(function(d){return d;});
</script>  

__3. Enter, Update and Exit__ 
-----------------------------  

The example shown above is a perfect one that the data would bind with _DOM_ elements one by one exactly. In reality, however, the data would be more than the _DOM_ elements which it is going to bind, or less. This is where the _Enter_, _Update_ and _Exit_ come to play.  

Let's start with the case which the number of _DOM_ elements is fewer than the data. We need _Enter_ to create more _DOM_ to fit the data. For example, if we first have only six symbols, and the _label_ has seven elements. We use the following code to fullfil the task of _Enter_.  

```javascript  
var label1 = [4/3,2,3,4,5,6,7];
svg.selectAll("path")
   .data(label1)
   .enter()
   .append("path")
   .attr("transform", function(d,i) { return "translate("+margin.left*(d-1)*3+","+margin.top+")";})
   .attr('d', d3.symbol().type(function(d,i) { return d3.symbols[i];}).size(function(d,i) {return 500*d}))
   .attr("fill", d=>color(d));
```  

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#example5 {
   background-color: lightgrey;
}

</style>

<svg id="example5" width="950" height="150"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script>
var margin = {top: 60, right: 50, bottom: 30, left: 50};
var width = document.getElementById("example5").getBoundingClientRect().width-100;
var height = 150;
var label = [4/3,2,3,4,5,6];
var color = d3.scaleOrdinal(d3.schemeCategory10);
    
var svg = d3.select("#example5")
            .append("g")
            .attr("width",width)
            .attr("height",height);

svg.selectAll("path")
   .data(label)
   .enter()
   .append("path")
   .attr("transform", function(d,i) { return "translate("+margin.left*(d-1)*3+","+margin.top+")";})
   .attr('d', d3.symbol().type(function(d,i) { return d3.symbols[i];}).size(function(d,i) {return 500*d}))
   .attr("fill", d=>color(d));

var text = [1,2,3,4,5,6,7];
d3.select("#example5")
  .selectAll("text")
  .data(text)
  .enter()
  .append("text")
  .attr("fill","red")
  .attr("font-size","30")
  .attr("x",function(d,i){
                          if(i==0) {return margin.left-10;}
                          else {return margin.left*3*(d-1)-10;}
                          })
  .attr("y",140)
  .text(function(d){return d;});

var label1 = [4/3,2,3,4,5,6,7];
svg.selectAll("path")
   .data(label1)
   .enter()
   .append("path")
   .attr("transform", function(d,i) { return "translate("+margin.left*(d-1)*3+","+margin.top+")";})
   .attr('d', d3.symbol().type(function(d,i) { return d3.symbols[i];}).size(function(d,i) {return 500*d}))
   .attr("fill", d=>color(d));
</script>  

By opening browser console, we will find a different piece of information compared with above:  

![alt text](/img/blog/d3_selection5.png) 

And when we have more data than _DOM_ elements, we need _Exit_ to remove redundant _DOM_ to match with the data. Here is the code to use:    

```javascript  
var label1 = [4/3,2,3,4,5,6];
svg.selectAll("path")
   .data(label1)
   .exit()
   .remove()
   .append("path")
   .attr("transform", function(d,i) { return "translate("+margin.left*(d-1)*3+","+margin.top+")";})
   .attr('d', d3.symbol().type(function(d,i) { return d3.symbols[i];}).size(function(d,i) {return 500*d}))
   .attr("fill", d=>color(d));
```  

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#example6 {
   background-color: lightgrey;
}

</style>

<svg id="example6" width="950" height="150"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script>
var margin = {top: 60, right: 50, bottom: 30, left: 50};
var width = document.getElementById("example6").getBoundingClientRect().width-100;
var height = 150;
var label = [4/3,2,3,4,5,6,7];
var color = d3.scaleOrdinal(d3.schemeCategory10);
    
var svg = d3.select("#example6")
            .append("g")
            .attr("width",width)
            .attr("height",height);

svg.selectAll("path")
   .data(label)
   .enter()
   .append("path")
   .attr("transform", function(d,i) { return "translate("+margin.left*(d-1)*3+","+margin.top+")";})
   .attr('d', d3.symbol().type(function(d,i) { return d3.symbols[i];}).size(function(d,i) {return 500*d}))
   .attr("fill", d=>color(d));

var text = [1,2,3,4,5,6,7];
d3.select("#example6")
  .selectAll("text")
  .data(text)
  .enter()
  .append("text")
  .attr("fill","red")
  .attr("font-size","30")
  .attr("x",function(d,i){
                          if(i==0) {return margin.left-10;}
                          else {return margin.left*3*(d-1)-10;}
                          })
  .attr("y",140)
  .text(function(d){return d;});

var label1 = [4/3,2,3,4,5,6];
svg.selectAll("path")
   .data(label1)
   .exit()
   .remove()
   .append("path")
   .attr("transform", function(d,i) { return "translate("+margin.left*(d-1)*3+","+margin.top+")";})
   .attr('d', d3.symbol().type(function(d,i) { return d3.symbols[i];}).size(function(d,i) {return 500*d}))
   .attr("fill", d=>color(d));
</script>  

After introducing _Enter_ and _Exit_, it would be quite easy to understand _Update_. It is basically using the _data()_ to update _DOM_ elements if the data is updated.   

<br />

<u>Reference:</u>    
----------------- 
(1). D3 API Reference, _[https://github.com/d3/d3/blob/master/API.md](https://github.com/d3/d3/blob/master/API.md)_.  
(2). d3-selection API, _[https://github.com/d3/d3-selection](https://github.com/d3/d3-selection)_.   
(3). D3's enter() and exit(): Under the Hood, _[http://animateddata.co.uk/lab/d3enterexit/](http://animateddata.co.uk/lab/d3enterexit/)_.  
(4). How Selections Work, _[https://bost.ocks.org/mike/selection/](https://bost.ocks.org/mike/selection/)_.  
(5). Thinking with Joins, _[https://bost.ocks.org/mike/join/](https://bost.ocks.org/mike/join/)_.  

[1]: https://d3js.org/  "d3"
