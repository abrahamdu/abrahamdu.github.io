---
layout: project
section-type: project
title: Maps
sitemap:
  priority: 1.0
---  
__United States__

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

<svg id="state" width="1200" height="900"></svg>
<div class="tooltip"></div>
<script src="https://d3js.org/d3.v4.min.js"></script>
<script src="https://d3js.org/topojson.v2.min.js"></script>
<script src="https://d3js.org/d3-queue.v3.min.js"></script>
<script>
var margin = {top: 10, right: 10, bottom: 10, left: 10};
var width = document.getElementById("state").getBoundingClientRect().width;
var height = 900;
var path = d3.geoPath();
var svg = d3.select("#state")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
var tooltip = d3.select("div.tooltip");
d3.queue()
  .defer(d3.json, "/project/maps/data/us_2015.json")
  .defer(d3.csv, "/project/maps/data/us-states-names.csv")
  .await(ready);
function ready(error, us, names) {
  if (error) throw error;
  var states = topojson.feature(us, us.objects.states).features;
    states_name = states.filter(function(d) {
    return names.some(function(n) {
      if (d.id == n.id) return d.name = n.State;
    })});
  svg.selectAll("path")
			.data(states_name)
			.enter()
			.append("path")
			.attr("stroke","grey")
			.attr("stroke-width",1)
            .attr("fill", "white")
			.attr("d", path )
            .on("mouseover",function(d,i){
                d3.select(this).attr("stroke-width",2);
                return tooltip.style("hidden", false).html(d.name);
            })
            .on("mousemove",function(d){
                tooltip.classed("hidden", false)
                       .style("top", (d3.event.pageY - 250) + "px")
                       .style("left", (d3.event.pageX - 170) + "px")
                       .html(d.name);
            })
            .on("mouseout",function(d,i){
                d3.select(this).attr("stroke-width",1);
                tooltip.classed("hidden", true);
            });
};
</script>   
