---
layout: post
section-type: post
published: true
category: VISUALIZATION
tags: [ 'd3', 'javascript', 'json' ]
title: 'Geographic Data for D3 - from GeoJSON to TopoJSON'
date: 2018-01-08
modifiedOn: 2018-01-08
---  

During this post, I will go through from the basics of [GeoJSON] [2] and [TopoJSON] [3] to comparing the difference and improvement from one to another and finally use simple examples to illustrate how to optimize the size of [TopoJSON] [3] by _Quantizing_ and _Simplying_ without losing the quality of data visualization.  

__1. What is GeoJSON?__  
-----------------------  
Based on 2015 IETF, _the Internet Engineering Task Force_, [GeoJSON] [2] is defined as a JSON format for encoding data about geographic features. [GeoJSON] [2] could represent a region of space (a _Geometry_), a spatially bounded entity (a _Feature_), or a list of Features (a _FeatureCollection_). [GeoJSON] [2] supports the following geometry types: Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, and GeometryCollection. Features in [GeoJSON] [2] contain a Geometry object and additional properties, and a FeatureCollection contains a list of Features. A _Feature_ consists of _Geometry_ and additional elements and a _FeatureCollection_ is just an array of _Feature_ objects.    

__1.1 Geometry__  
----------------  
A _Geometry_ object consists of a _type_ and a collection of coordinates which defines the position of subject of _type_. The components start with simple units: _Point_ for one dimension, _LineString_ for two dimensions, and  _Polygon_ for three dimensions. The complications of [GeoJSON] [2] are all based on any of these three types.    

__Point__  

_Point_ is just a simple point defined by its coordinates of position by the convention order longitude and latitude.  

```json  
{ "type": "Point", "coordinates": [0, 0] }  
```  
__LineString__   

_LineString_ is the line with starting point and ending point.  

```json
{ "type": "LineString", "coordinates": [[0, 0], [10, 10]] }
```  
__Polygon__  

_Polygon_ is more complicated than _Point_ and _LineString_ since it has shapes. There are two types of _Polygons_. One comes without holes.   

```json  
{
  "type": "Polygon",
  "coordinates": [
    [
      [0, 0], [10, 10], [10, 0], [0, 0]
    ]
  ]
}
```  

And the other comes with holes.  

```json  
{
   "type": "Polygon",
   "coordinates": [
       [ [100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0] ], // exterior boundary
       [ [100.2, 0.2], [100.8, 0.2], [100.8, 0.8], [100.2, 0.8], [100.2, 0.2] ]  // interior boundary
   ]
}
```  
On top of these three basic units, we have three extensions of each type by adding multiples onto each type.  

__MultiPoint__    

An array of _Point_ objects.  

```json  
{
   "type": "MultiPoint",
   "coordinates": [
       [100.0, 0.0], [101.0, 1.0]
   ]
}
```  

__MultiLineString__  

An array of _LineString_ objects.  

```json
{
   "type": "MultiLineString",
   "coordinates": [
       [ [100.0, 0.0], [101.0, 1.0] ],
       [ [102.0, 2.0], [103.0, 3.0] ]
   ]
}
```

__MultiPolygon__  

An array of _Polygon_ objects.  

```json  
{
   "type": "MultiPolygon",
   "coordinates": [
       [
           [ [102.0, 2.0], [103.0, 2.0], [103.0, 3.0], [102.0, 3.0], [102.0, 2.0] ]
       ],
       [
           [ [100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0] ],
           [ [100.2, 0.2], [100.8, 0.2], [100.8, 0.8], [100.2, 0.8], [100.2, 0.2] ]
       ]
   ]
}
```  
__GeometryCollection__  
The above six types of geometry could be combined together to create _GeometryCollection_.  

```json  
{ "type": "GeometryCollection",
    "geometries": [
      { "type": "Point",
        "coordinates": [100.0, 0.0]
        },
      { "type": "LineString",
        "coordinates": [ [101.0, 0.0], [102.0, 1.0] ]
        }
    ]
}
```  
All the seven types of Geometries, Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, and GeometryCollection, are case-sensitive. The order convension of coordinates follow the longitude-latitude-elevation order.  

__1.2 Feature__  
---------------  
A _Feature_ is an object of collection of geometry and additional properties and both geometry and properties are required by _Feature_. Specifically, _Feature_ will have _type_ property with value _Feature_, _geometry_ property as well as _properties_ property.  

```json  
{
   "type": "Feature",
   "geometry": {
       "type": "LineString",
       "coordinates": [
           [100.0, 0.0], [101.0, 1.0]
       ]
   },
   "properties": {
       "prop0": "value0",
       "prop1": "value1"
   }
}
```  

__1.3 FeatureCollection__
-------------------------  
Not surprisingly, _FeatureCollection_ is just an array of _Feature_ which has _type_ property with value _FeatureCollection_ and _features_.  

```json  
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [0, 0]
      },
      "properties": {
        "name": "null island"
      }
    }
  ]
}  
```  

__1.4 Bounding Box__  
--------------------  
[GeoJSON] [2] may have a member called "bbox", bounding box which contains information on the coordinate range for its geometries, features or featurecollections. It follows the convension of longitude-latitude-elevation min-max order going from left, bottom, right to top counter-clockwise which defines the boundary of underlying geo-information.  

```json  
{
       "type": "Feature",
       "bbox": [-10.0, -10.0, 10.0, 10.0],
       "geometry": {
           "type": "Polygon",
           "coordinates": [
               [
                   [-10.0, -10.0],
                   [10.0, -10.0],
                   [10.0, 10.0],
                   [-10.0, -10.0]
               ]
           ]
       }
}
```  

__2. TopoJSON__  
---------------  
[TopoJSON] [3] is an extension of [GeoJSON] [2] which eliminates redundancy to allow geometries to be stored more efficiently. 

According to [TopoJSON Format Specification](https://github.com/topojson/topojson-specification), it must contain a "type" member, usually "Topology", a "objects" member, itself another object named "example". Geometry object _Point_ and _MultiPoint_ must have a "coordinates" member while _LineString_, _Polygon_, _MultiLineString_ and _MultiPolygon_ must have a "arcs" memeber. Both "coordinates" and "arcs" are always an array. "bbox" is optional as well as "transform" which is used to construct "quantized" topology. I use the simple examples in the [GeoJSON] [2] session to illustrate [TopoJSON] [3].    

```json  
//Point
{"type":"Topology","objects":{"example":{"type":"Point","coordinates":[0,0]}},"arcs":[],"bbox":[0,0,0,0]}

//LineString
{"type":"Topology","objects":{"example":{"type":"LineString","arcs":[0]}},"arcs":[[[0,0],[10,10]]],"bbox":[0,0,10,10]}

//Polygon
{"type":"Topology","objects":{"example":{"type":"Polygon","arcs":[[0]]}},"arcs":[[[0,0],[10,10],[10,0],[0,0]]],"bbox":[0,0,10,10]}

//MultiPoint
{"type":"Topology","objects":{"example":{"type":"MultiPoint","coordinates":[[100,0],[101,1]]}},"arcs":[],"bbox":[100,0,101,1]}

//MultiLineString
{"type":"Topology","objects":{"example":{"type":"MultiLineString","arcs":[[0],[1]]}},"arcs":[[[100,0],[101,1]],[[102,2],[103,3]]],"bbox":[100,0,103,3]}

//MultiPolygon
{"type":"Topology","objects":{"example":{"type":"MultiPolygon","arcs":[[[0]],[[1],[2]]]}},"arcs":[[[102,2],[103,2],[103,3],[102,3],[102,2]],[[100,0],[101,0],[101,1],[100,1],[100,0]],[[100.2,0.2],[100.8,0.2],[100.8,0.8],[100.2,0.8],[100.2,0.2]]],"bbox":[100,0,103,3]}

//GeometryCollection
{"type":"Topology","objects":{"example":{"type":"GeometryCollection","geometries":[{"type":"Point","coordinates":[100,0]},{"type":"LineString","arcs":[0]}]}},"arcs":[[[101,0],[102,1]]],"bbox":[100,0,102,1]}

//Feature
{"type":"Topology","objects":{"example":{"type":"LineString","arcs":[0],"properties":{"prop0":"value0","prop1":"value1"}}},"arcs":[[[100,0],[101,1]]],"bbox":[100,0,101,1]}

//FeatureCollection
{"type":"Topology","objects":{"example":{"type":"GeometryCollection","geometries":[{"type":"Point","coordinates":[0,0],"properties":{"name":"null island"}}]}},"arcs":[],"bbox":[0,0,0,0]}
```  
As we can find out, all [TopoJSON] [3] counterparties have a "type" member with value "Topology". The topology objects are all with "example" object and the differences start with it by different types of geometries. For _Point_ and _MultiPoint_, they have both "coordinates" and "arcs" members although "arcs" is always null since the position information is carried over by "coordinates" while the rest _LineString_, _Polygon_, _MultiLineString_ and _MultiPolygon_ only have "arcs" member.  

__3. From Raw Data to TopoJSON__  
--------------------------------  
In reality, we need to create our own [TopoJSON] [3] file for [D3] [1]'s consumption from raw ShapeFile formats. I will go through steps borrowed from Bostock's series of blogs [1](https://medium.com/@mbostock/command-line-cartography-part-1-897aa8f8ca2c), [2](https://medium.com/@mbostock/command-line-cartography-part-2-c3a82c5c0f3), [3](https://medium.com/@mbostock/command-line-cartography-part-3-1158e4c55a1e) and [4](https://medium.com/@mbostock/command-line-cartography-part-4-82d0d26df0cf), and Ændrew Rininsland's [another view](https://medium.com/@aendrew/creating-topojson-using-d3-v4-10838d1a9538).  

To start with, we need install packages needed for data manipulation, which are __shapefile__ for converting ShapeFile to [GeoJSON] [2], and __topojson__ for converting [GeoJSON] [2] to [TopoJSON] [3].  

```javascript  
npm install -g shapefile ndjson topojson ndjson-cli
```
I used US Census Bureau published [2016 States Shapefiles](https://www.census.gov/geo/maps-data/data/cbf/cbf_state.html) and unzip it into my local directory.  

```javascript  
shp2json cb_2016_us_state_5m.shp -o cb_2016_us_state_5m.json
geo2topo cb_2016_us_state_5m.json > cb_2016_us_state_5m.topo.json
```  
For just a quick check, the above two commands would suffice to convert raw shapefiles into [TopoJSON] [3] file. If you check the size of each file, it is not hart to find out the [TopoJSON] [3] is only about 70% of original [GeoJSON] [2] file.  

![alt text](/img/blog/geojson1.png)  

Usually, it is not optimal to take advantage of [TopoJSON] [3]'s capability to meet different particular needs for [D3] [1]. We will deep dive to test a few ways of optimizing the file convension.  

First of all, we convert the raw data into newline-delimited features with one feature per line for human-beings easy to read and let us to use convenient __ndjson-cli__ tool.   

To start with, we first rely on the newline-delimited file to convert into [TopoJSON] [3] for benchmarking.  

```javascript  
shp2json -n cb_2016_us_state_5m.shp > cb_2016_us_state_5m.ndjson
geo2topo -n cb_2016_us_state_5m.ndjson > cb_2016_us_state_5m.topo1.json
```  
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
</style>
<h3>Benchmarking TopoJSON:</h3>
<svg id="topo1" width="1200" height="700"></svg>
<script src="https://d3js.org/d3.v4.min.js"></script>
<script src="https://d3js.org/topojson.v2.min.js"></script>
<script src="https://d3js.org/d3-queue.v3.min.js"></script>
<script>
var margin = {top: 10, right: 10, bottom: 10, left: 10};
var width = document.getElementById("topo1").getBoundingClientRect().width;
var height = 700;
var svg1 = d3.select("#topo1")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
var projection = d3.geoAlbersUsa();
var path = d3.geoPath()
             .projection(projection);
d3.queue()
  .defer(d3.json, "/blog/data/cb_2016_us_state_5m.topo1.json")
  .await(ready);
function ready(error, us1) {
  if (error) throw error;
  var topo1 = topojson.feature(us1, us1.objects.cb_2016_us_state_5m).features;
  svg1.append("g")
            .attr("stroke","grey")
			.attr("stroke-width",1)
            .attr("fill","white")
            .selectAll("path")
            .data(topo1)
            .enter()
			.append("path")
			.attr("d", path); 
    svg1.append("g")
       .attr("class", "state-borders")
       .selectAll("path")
       .data(topo1)
       .enter()
       .append("path")
       .attr("d", path);
};
</script>   

Then, we can take this benchmarking [TopoJSON] [3] file by quantizing and simplying. 

_Quantizing_ is basically reducing coordinate precision. It is implemented by _topoquantize_ with option as numbers. Indicated by [TopoJSON] [3] [API](topoquantize 1e5 < cb_2016_us_state_5m.topo1.json > cb_2016_us_state_5m.topo2.json), it is typically powers of ten. The bigger number is, the more precise.  

```javascript  
topoquantize 1e5 < cb_2016_us_state_5m.topo1.json > cb_2016_us_state_5m.topo2.json
```  
<h3>Quantizing:</h3>
<svg id="topo2" width="1200" height="700"></svg>
<script src="https://d3js.org/d3.v4.min.js"></script>
<script src="https://d3js.org/topojson.v2.min.js"></script>
<script src="https://d3js.org/d3-queue.v3.min.js"></script>
<script>
var margin = {top: 10, right: 10, bottom: 10, left: 10};
var width = document.getElementById("topo2").getBoundingClientRect().width;
var height = 700;
var svg2 = d3.select("#topo2")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
d3.queue()
  .defer(d3.json, "/blog/data/cb_2016_us_state_5m.topo2.json")
  .await(ready);
function ready(error, us2) {
  if (error) throw error;
  var topo2 = topojson.feature(us2, us2.objects.cb_2016_us_state_5m).features;
  svg2.append("g")
            .attr("stroke","grey")
			.attr("stroke-width",1)
            .attr("fill","white")
            .selectAll("path")
            .data(topo2)
            .enter()
			.append("path")
			.attr("d", path); 
    svg2.append("g")
       .attr("class", "state-borders")
       .selectAll("path")
       .data(topo2)
       .enter()
       .append("path")
       .attr("d", path);
};
</script>   

_Simplying_ is basically reducing the number of nodes used to represent arcs. It is implemented by _toposimplify_ by _-p_ option. Opposite from _topoquantize_, the value should be from 0 to 1 and the smaller it is, the more precise. _f_ just removes detached rings that are smaller than the simplification threshold after simplifying.  

```javascript
toposimplify -p 1e-1 -f < cb_2016_us_state_5m.topo2.json > cb_2016_us_state_5m.topo3.json
```  
<h4>Simplying:</h4>
<svg id="topo3" width="1200" height="700"></svg>
<script src="https://d3js.org/d3.v4.min.js"></script>
<script src="https://d3js.org/topojson.v2.min.js"></script>
<script src="https://d3js.org/d3-queue.v3.min.js"></script>
<script>
var margin = {top: 10, right: 10, bottom: 10, left: 10};
var width = document.getElementById("topo3").getBoundingClientRect().width;
var height = 700;
var svg3 = d3.select("#topo3")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
d3.queue()
  .defer(d3.json, "/blog/data/cb_2016_us_state_5m.topo3.json")
  .await(ready);
function ready(error, us3) {
  if (error) throw error;
  var topo3 = topojson.feature(us3, us3.objects.cb_2016_us_state_5m).features;
  svg3.append("g")
            .attr("stroke","grey")
			.attr("stroke-width",1)
            .attr("fill","white")
            .selectAll("path")
            .data(topo3)
            .enter()
			.append("path")
			.attr("d", path); 
    svg3.append("g")
       .attr("class", "state-borders")
       .selectAll("path")
       .data(topo3)
       .enter()
       .append("path")
       .attr("d", path);
};
</script>   

The size of each data conversion is as follows:  

![alt text](/img/blog/geojson2.png)  

It is not hard to discover that by _Quantizing_ the file, not only does the file size decrease tremendously for fast rendering, but also the quality of visualization is kept.  

<br />

<u>Reference:</u>    
----------------- 
(1). TopoJSON API, _[https://github.com/topojson/topojson](https://github.com/topojson/topojson)_.  
(2). The GeoJSON Specification (RFC 7946), _[https://tools.ietf.org/html/rfc7946](https://tools.ietf.org/html/rfc7946)_.  
(3). More than you ever wanted to know about GeoJSON, _[https://macwright.org/2015/03/23/geojson-second-bite](https://macwright.org/2015/03/23/geojson-second-bite)_.    
(4). The TopoJSON Format Specification, _[https://github.com/topojson/topojson-specification](https://github.com/topojson/topojson-specification)_.  
(5). How To Infer Topology, _[https://bost.ocks.org/mike/topology/](https://bost.ocks.org/mike/topology/)_.  
(6). Spatial data on a diet: tips for file size reduction using TopoJSON, _[http://zevross.com/blog/2014/04/22/spatial-data-on-a-diet-tips-for-file-size-reduction-using-topojson/](http://zevross.com/blog/2014/04/22/spatial-data-on-a-diet-tips-for-file-size-reduction-using-topojson/)_.  

[1]: https://d3js.org/  "d3"
[2]: http://geojson.org/ "geojson"
[3]: https://github.com/topojson/topojson "topojson"