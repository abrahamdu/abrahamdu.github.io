---
layout: post
section-type: post
published: true
category: VISUALIZATION
tags: [ 'd3', 'html', 'css', 'javascript' ]
title: 'D3 Cluster Diagram'
date: 2017-07-06
modifiedOn: 2017-07-06
---  

The previous series of posts show different layouts of charts with no obvious relationships between components. In reality, there are situations where relational connections between components need to be shown, such as [family tree](https://www.ancestry.com/family-tree/) and [organizational chart](https://en.wikipedia.org/wiki/Organizational_chart). [D3] [1] implements all these hierarchical relationship in the [d3-hierarchy](https://github.com/d3/d3-hierarchy) module.  

The first hierarchy chart to show is the cluster layout which, in [D3] [1]'s API, produces [dendrogram](https://github.com/d3/d3-hierarchy). I take advantage of [Curran](https://github.com/curran/data)'s [UN International migrate stock 2015](https://github.com/curran/data/blob/gh-pages/un/placeHierarchy/countryHierarchy.json) data to get the list of countries at the level of continent and region in order to show the hierarchy.  

Similar as the other layouts we go through previously, we start with cluster layout:  

```javascript  
var cluster = d3.cluster()
	.size([height, width - 140]);
```  
Since the data I use is already in a hierarchical format, I can directly use _hierarchy()_ function to construct a root node which is "world" in this particular example and later pass this to the _cluster_ created before.  

```javascript  
var hierarchy = d3.hierarchy(data);
cluster(hierarchy);
```  

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#world_cluster {
   background-color: lightgrey;
}

.node circle {
  fill: #999;
}

.node text {
  font: 10px sans-serif;
}

.node--internal circle {
  fill: #555;
}

.node--internal text {
  text-shadow: 0 1px 0 #fff, 0 -1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff;
}

.link {
  fill: none;
  stroke: #555;
  stroke-opacity: 0.4;
  stroke-width: 1.5px;
}
</style>

<svg id="world_cluster" width="950" height="2100"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script>
var margin = {top: 30, right: 50, bottom: 30, left: 50};
var width = document.getElementById("world_cluster").getBoundingClientRect().width-100;
var height = 2000;

var g = d3.select("#world_cluster")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var cluster = d3.cluster()
	.size([height, width - 140]);
    
d3.json("/blog/data/all_countries_2015.json", function(error, data) {
  if (error) throw error;
  var hierarchy = d3.hierarchy(data);
  cluster(hierarchy);
  
  var link = g.selectAll(".link")
      .data(hierarchy.descendants().slice(1))
      .enter().append("path")
      .attr("class", "link")
      .attr("d", function diagonal(d) {
          if(d.parent === hierarchy.descendants[0]){
          return "M" + d.y + "," + d.x
          + " " + d.parent.y + "," + d.parent.x
          } else {
          return "M" + d.y + "," + d.x
          + "C" + (d.parent.y + 100) + "," + d.x
          + " " + (d.parent.y + 100) + "," + d.parent.x
          + " " + d.parent.y + "," + d.parent.x;
          }
          }
       );

   var node = g.selectAll(".node")
      .data(hierarchy.descendants())
      .enter().append("g")
      .attr("class", function(d) { return "node" + (d.children ? " node--internal" : " node--leaf");})
      .attr("transform", function(d) {
        return "translate(" + d.y + "," + d.x + ")"; 
      })

  node.append("circle")
      .attr("r", 3);
  
  node.append("text")
      .attr("dy", 3)
      .attr("x", function(d) { return d.children ? -6 : 6; })
      .style("text-anchor", function(d) { return d.children ? "end" : "start"; })
      .text(function(d) { 
        return d.data.data.id;
      });
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

#world_cluster {
   background-color: lightgrey;
}

.node circle {
  fill: #999;
}

.node text {
  font: 10px sans-serif;
}

.node--internal circle {
  fill: #555;
}

.node--internal text {
  text-shadow: 0 1px 0 #fff, 0 -1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff;
}

.link {
  fill: none;
  stroke: #555;
  stroke-opacity: 0.4;
  stroke-width: 1.5px;
}
</style>

<svg id="world_cluster" width="950" height="2100"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script>
var margin = {top: 30, right: 50, bottom: 30, left: 50};
var width = document.getElementById("world_cluster").getBoundingClientRect().width-100;
var height = 2000;

var g = d3.select("#world_cluster")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var cluster = d3.cluster()
	.size([height, width - 140]);
    
d3.json("/blog/data/all_countries_2015.json", function(error, data) {
  if (error) throw error;
  var hierarchy = d3.hierarchy(data);
  cluster(hierarchy);
  
  var link = g.selectAll(".link")
      .data(hierarchy.descendants().slice(1))
      .enter().append("path")
      .attr("class", "link")
      .attr("d", function diagonal(d) {
          if(d.parent === hierarchy.descendants[0]){
          return "M" + d.y + "," + d.x
          + " " + d.parent.y + "," + d.parent.x
          } else {
          return "M" + d.y + "," + d.x
          + "C" + (d.parent.y + 100) + "," + d.x
          + " " + (d.parent.y + 100) + "," + d.parent.x
          + " " + d.parent.y + "," + d.parent.x;
          }
          }
       );

   var node = g.selectAll(".node")
      .data(hierarchy.descendants())
      .enter().append("g")
      .attr("class", function(d) { return "node" + (d.children ? " node--internal" : " node--leaf");})
      .attr("transform", function(d) {
        return "translate(" + d.y + "," + d.x + ")"; 
      })

  node.append("circle")
      .attr("r", 3);
  
  node.append("text")
      .attr("dy", 3)
      .attr("x", function(d) { return d.children ? -6 : 6; })
      .style("text-anchor", function(d) { return d.children ? "end" : "start"; })
      .text(function(d) { 
        return d.data.data.id;
      });
});

</script>
```

<u>Reference:</u>    
----------------- 
(1). D3 API Reference, _[https://github.com/d3/d3/blob/master/API.md](https://github.com/d3/d3/blob/master/API.md)_.  
(2). D3 Hierarchy API, _[https://github.com/d3/d3-hierarchy](https://github.com/d3/d3-hierarchy)_.  
(3). Tidy Tree vs. Dendrogram, _[https://bl.ocks.org/mbostock/e9ba78a2c1070980d1b530800ce7fa2b](https://bl.ocks.org/mbostock/e9ba78a2c1070980d1b530800ce7fa2b)_.  
(4). World Countries Hierarchy, _[https://bl.ocks.org/curran/1dd7ab046a4ed32380b21e81a38447aa](https://bl.ocks.org/curran/1dd7ab046a4ed32380b21e81a38447aa)_.  

[1]: https://d3js.org/  "d3"