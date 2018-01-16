---
layout: project
section-type: project
title: Maps
sitemap:
  priority: 1.0
---  
<h1>United States</h1>

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}
.state-borders {
  fill: none;
  stroke: steelblue;
  stroke-width: 2;
}
.county :hover {
  fill: red;
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
      opacity: 0.8; 
      position: absolute;
}
</style>
<h2>State and County</h2>
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
  .defer(d3.csv, "/project/maps/data/us-county-names.csv")
  .await(ready);
function ready(error, us, st_names, county_names) {
  if (error) throw error;
  var states = topojson.feature(us, us.objects.states).features;
  states_name = states.filter(function(d) {
    return st_names.some(function(n) {
      if (d.id == n.id) return d.name = n.State;
    })});
  var counties = topojson.feature(us, us.objects.counties).features;
  counties_name = counties.filter(function(d) {
    return county_names.some(function(n) {
      if (d.id == n.statefp.concat(n.countyfp)) {return d.name = [n.countyname,n.state].join(',');}
    })});
  svg.append("g")
            .attr("stroke","grey")
			.attr("stroke-width",1)
            .attr("fill","white")
            .selectAll("path")
            .data(counties_name)
            .enter()
			.append("path")
			.attr("d", path)
            .on("mouseover",function(d,i){
                d3.select(this).attr("stroke-width",3).attr("fill", "red");
                return tooltip.style("hidden", false).html(d.name);
            })
            .on("mousemove",function(d){
                tooltip.classed("hidden", false)
                       .style("top", (d3.event.pageY - 250) + "px")
                       .style("left", (d3.event.pageX - 170) + "px")
                       .html(d.name);
            })
            .on("mouseout",function(d,i){
                d3.select(this).attr("stroke-width",1).attr("fill","white");
                tooltip.classed("hidden", true);
            }); 
    svg.append("g")
       .attr("class", "state-borders")
       .selectAll("path")
       .data(states_name)
       .enter()
       .append("path")
       .attr("d", path);
};
</script>   

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}
.state-borders {
  fill: none;
  stroke: steelblue;
  stroke-width: 2;
}
.cd :hover {
  fill: red;
}
.hidden {
      display: none;
}
div.tooltip_cd {
      color: #222; 
      background: #fff; 
      border-radius: 3px; 
      box-shadow: 0px 0px 2px 0px #a6a6a6; 
      padding: .2em; 
      text-shadow: #f5f5f5 0 1px 0;
      opacity: 0.8; 
      position: absolute;
}
</style>
<h2>State and Congressional District</h2>
<svg id="cd" width="1200" height="900"></svg>
<div class="tooltip_cd"></div>
<script src="https://d3js.org/d3.v4.min.js"></script>
<script src="https://d3js.org/topojson.v2.min.js"></script>
<script src="https://d3js.org/d3-queue.v3.min.js"></script>
<script>
var margin = {top: 10, right: 10, bottom: 10, left: 10};
var width = document.getElementById("cd").getBoundingClientRect().width;
var height = 900;
var projection = d3.geoAlbersUsa().scale(1280).translate([width / 2, height / 3]);
var path1 = d3.geoPath().projection(projection);
var svg1 = d3.select("#cd")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
var tooltip_cd = d3.select("div.tooltip_cd");
d3.queue()
  .defer(d3.json, "/project/maps/data/cb_2016_us_state_5m.json")
  .defer(d3.json, "/project/maps/data/cb_2016_us_cd115_5m.json")
  .defer(d3.csv, "/project/maps/data/us-states-names.csv")
  .await(ready1);
function ready1(error, us, cd, st_names) {
  if (error) throw error;
  var states = us.features;
  states_name = states.filter(function(d) {
    return st_names.some(function(n) {
      if (d.properties.STATEFP == n.id) return d.name = n.State;
    })});
  var cd1 = cd.features;
  cd_name = cd1.filter(function(d) {
    return st_names.some(function(n) {
      if (d.properties.STATEFP == n.id) return d.name = [n.State,d.properties.CD115FP].join(' ');
    })});
    console.log(states,cd1,cd_name);
  svg1.append("g")
            .attr("stroke","grey")
			.attr("stroke-width",1)
            .attr("fill","white")
            .selectAll("path")
            .data(cd_name)
            .enter()
			.append("path")
			.attr("d", path1)
            .on("mouseover",function(d,i){
                d3.select(this).attr("stroke-width",3).attr("fill", "red");
                return tooltip_cd.style("hidden", false).html(d.name);
            })
            .on("mousemove",function(d){
                tooltip_cd.classed("hidden", false)
                       .style("top", (d3.event.pageY - 250) + "px")
                       .style("left", (d3.event.pageX - 170) + "px")
                       .html(d.name);
            })
            .on("mouseout",function(d,i){
                d3.select(this).attr("stroke-width",1).attr("fill","white");
                tooltip_cd.classed("hidden", true);
            }); 
    svg1.append("g")
       .attr("class", "state-borders")
       .selectAll("path")
       .data(states_name)
       .enter()
       .append("path")
       .attr("d", path1);
};
</script>   