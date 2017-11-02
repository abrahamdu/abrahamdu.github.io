---
layout: post
section-type: post
published: true
category: VISUALIZATION
tags: [ 'd3', 'html', 'css', 'javascript' ]
title: 'Read in Data by D3'
date: 2017-10-26
modifiedOn: 2017-10-26
---  

[D3] [1] stands for data-driven documents, which means data is the oxygen of final visualization. It is always true that a good visualization would have a solid foundation of data. [D3] [1] is not exception.  

More often than not, we prepare data and put it in some form of flat file as external source. [D3] [1] expands data support by module _d3-dsv_.  It serves as parser and formatter to delimiter-separated values, such as comma and tab. There are mainly three functions under this module: creator of parser/formatter, parser and formatter.  

```javascript
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script>
    csv1 = d3.dsvFormat(",").parse("last,first\nDu,Yi");
    csv2 = d3.dsvFormat(",").parseRows("last,first\nDu,Yi");
    tsv1 = d3.dsvFormat("\t").parse("last\tfirst\nDu\tYi");
    tsv2 = d3.dsvFormat("\t").parseRows("last\tfirst\nDu\tYi");
    console.log(csv1,typeof(csv1)); 
    console.log(csv2,typeof(csv2)); 
    console.log(tsv1,typeof(tsv1));
    console.log(tsv2,typeof(tsv2));
</script>
```  
From the example above, in order to parse some delimiter-separated values, a formatter needs to be defined first by using _d3.dsvFormat()_ function, which is used later to pass _parse_/_parseRows_ function for processing. The final parsing results always return an array. The only difference between _parse_ and _parseRows_ is that the former would return an object versus the latter would return rows.  

![alt text](/img/blog/d3-tsv1.png)  

The _parse_ function used above would also generate a new property for the arry named 'columns' as we can see from the console and we can invoke this property by using _data.columns_. This is also a new feature of [D3] [1] version 4 or above.   

```javascript  
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script>
    csv1 = d3.dsvFormat(",").parse("last,first\nDu,Yi");
    tsv1 = d3.dsvFormat("\t").parse("last\tfirst\nDu\tYi");
    console.log(csv1.columns); 
    console.log(tsv1.columns);
</script>
```  
![alt text](/img/blog/d3-tsv3.png)

From the other way around, we can use the same to create a formatter by _d3.dsvFormat_ function. In stead of using parse, _format_/_formatRows_ would be called to format an object to a string.  

```javascript  
<script src="https://d3js.org/d3-dsv.v1.min.js"></script>
<script>
    csv3 = d3.dsvFormat(",").format([{last: "Du", first: "Yi"}]);
    csv4 = d3.dsvFormat(",").formatRows([["last", "first"], ["Du", "Yi"]]);
    tsv3 = d3.dsvFormat("\t").format([{last: "Du", first: "Yi"}]);
    tsv4 = d3.dsvFormat("\t").formatRows([["last", "first"], ["Du", "Yi"]]);
    console.log(csv3,typeof(csv3)); 
    console.log(csv4,typeof(csv4));
    console.log(tsv3,typeof(tsv3));
    console.log(tsv4,typeof(tsv4));
</script>
```  
![alt text](/img/blog/d3-tsv2.png) 

For convenience, comma and tab delimiters are built-in function under this module. Here is a descriptive tree map to illustrate their relationships:  

<style>
svg {
    width: 100%;
    height: 100%;
    position: center;
}

#tree {
   background-color: black;
}

.link {
  fill: none;
  stroke: #ccc;
  stroke-width: 2px;
   }
   
.text-anchor { font-size: 22px; font-weight: bold; }

</style>

<svg id="tree" width="900" height="580"></svg>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script src="https://code.jquery.com/jquery-3.1.0.min.js"></script>

<script>
var tsv_tree =   {
                  "name": "d3.dsvFormat",
                  "value": "16",
                  "level": "steelblue",
                  "children": [
                              {
                              "name": "csv",
                              "value": "12",
                              "level": "red",
                              "children": [
                                          {
                                          "name": "parse",
                                          "value": "10",
                                          "level": "maroon",
                                          "children": [
                                                      {
                                                      "name": "d3.csvParse",
                                                      "value": "8",
                                                      "level": "maroon"
                                                      },
                                                      {
                                                      "name": "d3.csvParseRows",
                                                      "value": "8",
                                                      "level": "maroon"
                                                      }
                                                      ]
                                          },
                                          {
                                          "name": "format",
                                          "value": "10",
                                          "level": "pink",
                                          "children": [
                                                      {
                                                      "name": "d3.csvFormat",
                                                      "value": "8",
                                                      "level": "pink"
                                                      },
                                                      {
                                                      "name": "d3.csvFormatRows",
                                                      "value": "8",
                                                      "level": "pink"
                                                      }
                                                      ]
                                          }
                                          ]
                                },
                               {
                               "name": "tsv",
                               "value": "12",
                               "level": "gold",
                               "children": [
                                           {
                                           "name": "parse",
                                           "value": "10",
                                           "level": "orchid",
                                           "children": [
                                                       {
                                                      "name": "d3.tsvParse",
                                                      "value": "8",
                                                      "level": "orchid"
                                                      },
                                                      {
                                                      "name": "d3.tsvParseRows",
                                                      "value": "8",
                                                      "level": "orchid"
                                                      }
                                                       ]
                                           },
                                           {
                                           "name": "format",
                                           "value": "10",
                                           "level": "lime",
                                           "children": [
                                                        {
                                                      "name": "d3.tsvFormat",
                                                      "value": "8",
                                                      "level": "lime"
                                                      },
                                                      {
                                                      "name": "d3.tsvFormatRows",
                                                      "value": "8",
                                                      "level": "lime"
                                                      }
                                                       ]
                                          }
                                          ]
                                }
                                ]
                    };

var margin = {top: 30, right: 80, bottom: 30, left: 40};
var width = 900 - margin.left - margin.right;
var height = 900 - margin.top - margin.bottom;
    
var tree = d3.tree()
	.size([height - 300, width - 200]);
  
var nodes = d3.hierarchy(tsv_tree, function(d) {
    return d.children;
  });

nodes = tree(nodes);

var g = d3.select("#tree")
        .append("g")
        .attr("transform", "translate(" + margin.left*4 + "," + margin.top + ")");

var link = g.selectAll(".link")
      .data(nodes.descendants().slice(1))
      .enter().append("path")
      .attr("class", "link")
      .style("stroke", function(d) { return d.data.level; })
      .attr("d", function(d) {
       return "M" + d.y + "," + d.x
         + "C" + (d.y + d.parent.y) / 2 + "," + d.x
         + " " + (d.y + d.parent.y) / 2 + "," + d.parent.x
         + " " + d.parent.y + "," + d.parent.x;
       });
       
var node = g.selectAll(".node")
      .data(nodes.descendants())
      .enter().append("g")
      .attr("class", function(d) { return "node" + (d.children ? " node--internal" : " node--leaf");})
      .attr("transform", function(d) {
        return "translate(" + d.y + "," + d.x + ")"; 
      });
      
  node.append("circle")
      .attr("r", function(d) { return d.data.value; })
      .style("fill", function(d) { return d.data.level; });

  node.append("text")
      .attr("dy", 3)
      .attr("x", function(d) { return d.children ? (d.data.value ) * -1 : d.data.value})
      .text(function(d) { return d.data.name; })
      .style("text-anchor", function(d) { return d.children ? "end" : "start"; })
      .style("font-size", "22px")
      .attr("fill", "silver");
</script>

<u>Reference:</u>    
----------------- 
(1). D3 API Reference, _[https://github.com/d3/d3/blob/master/API.md](https://github.com/d3/d3/blob/master/API.md)_.  
(2). d3-tsv API, _[https://github.com/d3/d3-dsv](https://github.com/d3/d3-dsv)_.   

[1]: https://d3js.org/  "d3"