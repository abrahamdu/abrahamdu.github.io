---
layout: project
section-type: project
title: Maps
sitemap:
  priority: 1.0
---  
__China Map__

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}
.hidden {
      display: none;
}
div.tooltip {
      color: #222; 
      background: #fff; 
      border-radius: 3px; 
      box-shadow: 0px 0px 2px 0px #a6a6a6; 
      padding: .2em; 
      text-shadow: #f5f5f5 0 1px 0;
      opacity: 0.9; 
      position: absolute;
}
</style>

<svg id="china" width="1200" height="900"></svg>
<div class="tooltip"></div>
<script src="https://d3js.org/d3.v4.min.js"></script>
<script src="https://d3js.org/topojson.v2.min.js"></script>
<script src="https://d3js.org/d3-queue.v3.min.js"></script>
<script>
var margin = {top: 10, right: 10, bottom: 10, left: 10};
var width = document.getElementById("china").getBoundingClientRect().width;
var height = 900;
var color = d3.scaleOrdinal(d3.schemeCategory20c);
var projection = d3.geoMercator()
                   .center([110, 25]) 
                   .scale([800]) 
                   .translate([550,550])
                   .precision([.1]);
var path = d3.geoPath()
             .projection(projection);
var svg = d3.select("#china")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
var tooltip = d3.select("div.tooltip");
d3.json("/project/maps/data/china.geojson", function(error, china) {
  if (error) throw error;
  svg.selectAll("path")
			.data(china.features)
			.enter()
			.append("path")
			.attr("stroke","grey")
			.attr("stroke-width",1)
            .attr("fill", function(d,i){
				return color(i);
			})
			.attr("d", path )
			.on("mouseover",function(d,i){
                d3.select(this).attr("fill","grey").attr("stroke-width",2);
                return tooltip.style("hidden", false).html(d.properties.name);
            })
            .on("mousemove",function(d){
                tooltip.classed("hidden", false)
                       .style("top", (d3.event.pageY - 250) + "px")
                       .style("left", (d3.event.pageX - 170) + "px")
                       .html(d.properties.name);
            })
            .on("mouseout",function(d,i){
                d3.select(this).attr("fill",color(i)).attr("stroke-width",1);
                tooltip.classed("hidden", true);
            });
});
</script> 