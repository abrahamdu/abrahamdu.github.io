---
layout: project
section-type: project
title: Maps
sitemap:
  priority: 1.0
---  
__World Map__

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}
#world {
   background-color: #2B65EC;
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

<svg id="world" width="1200" height="900"></svg>
<div class="tooltip"></div>
<script src="https://d3js.org/d3.v4.min.js"></script>
<script src="https://d3js.org/topojson.v2.min.js"></script>
<script src="https://d3js.org/d3-queue.v3.min.js"></script>
<script>
var margin = {top: 10, right: 10, bottom: 10, left: 10};
var width = document.getElementById("world").getBoundingClientRect().width;
var height = 900;
var projection = d3.geoNaturalEarth1()
                   .center([0, 15]) 
                   .rotate([-9,0])
                   .scale([1300/(2*Math.PI)]) 
                   .translate([450,300]);
var path = d3.geoPath()
             .projection(projection);
var svg = d3.select("#world")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
var tooltip = d3.select("div.tooltip");
d3.queue()
  .defer(d3.json, "/project/maps/data/world-110m.json")
  .defer(d3.csv, "/project/maps/data/world-country-names.csv")
  .await(ready);
function ready(error, world, names) {
  if (error) throw error;
  var countries1 = topojson.feature(world, world.objects.countries).features;
    countries = countries1.filter(function(d) {
    return names.some(function(n) {
      if (d.id == n.id) return d.name = n.name;
    })});
  svg.selectAll("path")
			.data(countries)
			.enter()
			.append("path")
			.attr("stroke","green")
			.attr("stroke-width",1)
            .attr("fill", "white")
			.attr("d", path )
			.on("mouseover",function(d,i){
                d3.select(this).attr("fill","grey").attr("stroke-width",2);
                return tooltip.style("hidden", false).html(d.name);
            })
            .on("mousemove",function(d){
                tooltip.classed("hidden", false)
                       .style("top", (d3.event.pageY - 250) + "px")
                       .style("left", (d3.event.pageX - 170) + "px")
                       .html(d.name);
            })
            .on("mouseout",function(d,i){
                d3.select(this).attr("fill","white").attr("stroke-width",1);
                tooltip.classed("hidden", true);
            });
};
</script> 