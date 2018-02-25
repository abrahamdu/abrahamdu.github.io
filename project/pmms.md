---
layout: project
section-type: project
title: PMMS
sitemap:
  priority: 1.0
---  
<h1>PMMS - Primary Mortgage Market Survey by Freddie Mac</h1>

A historical view of [Freddie Mac PMMS](http://www.freddiemac.com/pmms/)

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#overall {
   background-color: lightgrey;
}

.axis--x path {
  fill: none;
  display: none;
}
</style>

<svg id="overall" width="950" height="600"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script src="https://d3js.org/topojson.v2.min.js"></script>
<script src="https://d3js.org/d3-queue.v3.min.js"></script>

<script>
var margin = {top: 30, right: 30, bottom: 30, left: 30};
var width = document.getElementById("overall").getBoundingClientRect().width-100;
var height = 500;
var parseTime = d3.timeParse("%m/%d/%Y");
var color = d3.scaleOrdinal(d3.schemeCategory10);
var g = d3.select("#overall")
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
var xRange = d3.scaleTime()
        .rangeRound([0, width]);
var yRange = d3.scaleLinear()
    .range([height, 0]);
var line = d3.line()
    .x(function(d) { return xRange(d.Date);  })
    .y(function(d) { return yRange(d.rate);  })
    .curve(d3.curveMonotoneX);
d3.csv("/project/data/pmms.csv",function(d, i, columns) {
    d.Date = parseTime(d.Week);
    for (var i = 1, n = columns.length, c; i < n; ++i) d[c = columns[i]] = +d[c];
    return d;
    },
    function(error,data){
    if (error) throw error;
    var overall = data.columns.slice(1,6).map(function(id) {
    return {
      id: id,
      values: data.map(function(d) {
      return {Date: d.Date, rate: d[id]};
      })
    };     
  });
  var final = new Array(overall[0],overall[2],overall[4]);
  xRange.domain(d3.extent(data, function(d) { return d.Date; }));
  yRange.domain([
    d3.min(final, function(c) { return d3.min(c.values, function(d) { return d.rate; }); }),
    d3.max(final, function(c) { return d3.max(c.values, function(d) { return d.rate; }); })
  ]);
color.domain(final.map(function(c) { return c.id; }));
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
      .text("% Rate");
var overalls = g.selectAll(".overalls")
    .data(final)
    .enter()
    .append("g")
    .attr("class", "overall");
  overalls.append("path")
      .style("fill", "none")
      .attr("d", function(d) { return line(d.values); })
      .style("stroke", function(d) { return color(d.id); });
  overalls.append("text")
      .datum(function(d) { return {id: d.id, value: d.values.slice(-1)[0]}; })
      .attr("transform", function(d) { return "translate(" + xRange(d.value.Date) + "," + yRange(d.value.rate) + ")"; })
      .attr("x", 3)
      .attr("dy", "0.35em")
      .style("font", "10px sans-serif")
      .text(function(d) { return d.id; })
      .style("stroke", function(d) { return color(d.id); });

});

</script>  

