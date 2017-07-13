---
layout: post
section-type: post
published: true
category: VISUALIZATION
tags: [ 'd3', 'html', 'css', 'javascript' ]
title: 'D3 Tree Diagram'
date: 2017-07-11
modifiedOn: 2017-07-11
---  

Similar as cluster, the tree layout by [D3] [1] produces tidy node-link diagram. In generally, tidy trees are more compact than clusters.  

Since both _cluster_ and _tree_ are under [d3-hierarchy](https://github.com/d3/d3-hierarchy) module, you would imagine that migration between each other is quite smooth.  

The only change to make is to evoke _tree()_ function instead of _cluster()_.  

```javascript   
var tree = d3.tree()
	.size([height, width - 140]);
```  
This is actually the only piece of code needed to be modified if you compare with [previous blog](http://anotherpeak.org/blog/visualization/2017/07/06/d3_clusterdiagram.html).  

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#world_tree {
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

<svg id="world_tree" width="950" height="2100"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script>
var margin = {top: 30, right: 50, bottom: 30, left: 50};
var width = document.getElementById("world_tree").getBoundingClientRect().width-100;
var height = 2000;

var g = d3.select("#world_tree")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var tree = d3.tree()
	.size([height, width - 140]);
    
d3.json("/blog/data/all_countries_2015.json", function(error, data) {
  if (error) throw error;
  var hierarchy = d3.hierarchy(data);
  tree(hierarchy);
  
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

One last thing I would like to mention is that when I first compare the __cluster__ and __tree__ diagram by exactly the same data, the difference is barely observed. After reading some documents especially the example in the [reference 2](https://bl.ocks.org/mbostock/e9ba78a2c1070980d1b530800ce7fa2b), I find this tiny difference at the bottom of each diagram.  

__Cluster__:  
![alt text](/img/blog/cluster_clip.png)    
__Tree__:  
![alt text](/img/blog/tree_clip.png) 

You can see _cluster_ is sort of right-aligned while _tree_ is left-aligned and because of this, _tree_ diagram could arrange the "children" nodes and links using blank spaces but _cluster_ couldn't take this advantage since the last-children nodes need to be arranged on the same column as you could see from above, which makes _tree_ more compact than _cluster_.  

<u>Reference:</u>    
----------------- 
(1). D3 API Reference, _[https://github.com/d3/d3/blob/master/API.md](https://github.com/d3/d3/blob/master/API.md)_.  
(2). Tidy Tree vs. Dendrogram, _[https://bl.ocks.org/mbostock/e9ba78a2c1070980d1b530800ce7fa2b](https://bl.ocks.org/mbostock/e9ba78a2c1070980d1b530800ce7fa2b)_. 

[1]: https://d3js.org/  "d3"