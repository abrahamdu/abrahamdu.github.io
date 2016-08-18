---
layout: post
section-type: post
published: true 
category: TECH
tags: [ 'json', 'javascript' ]
title: 'Understand JSON - Part 1: Introducing JSON'
date: 2016-01-25
modifiedOn: 2016-03-09
---

In the next couple of blogs, I will cover some basics of [JSON] [1] and how to parse it with different technical tools. And the goal of this series would help readers understand  [JSON] [1] better and use/parse it appropriately based on different scenarios.

![alt text](/img/blog/json.png)

[JSON] [1] stands for JavaScript Object Notation, first introduced to the world at JSON.org in 2001. It is a text format that facilitates structured data interchange between all programming languages. It is open, lightweight, and text-based data-interchange format. According to [The JSON Data Interchange Standard] [2], [JSON] [1] is syntax of braces, brackets, colons, and commas that is useful in many contexts, profiles, and applications. 

[JSON] [1] supports two widely-used data structures:
<li> A collection of name/value pairs</li>
<li> An ordered list of values</li>

[JSON] [1] supports the following data types:
<li> Null</li>
<li> True/False</li>
<li> Number</li>
<li> String</li>

[JSON] [1] uses the following structural tokens:
<li> Curly Brackets { }</li>
<li> Square Brackets [ ]</li>
<li> Colon :</li>
<li> Comma ,</li>

Combining data types with structural tokens, [JSON] [1] could generate different kinds of values: object, array, string, number, True/False and Null.
Object: a pair of curly bracket tokens surrounding zero or more name/value pairs. Name is a string and value could be any data types. A single colon is used to separate name from value and a single comma is used to separate different name/

```json
{"China": "86", "USA": "1"}
{
"ID": "F1234567",
"Company": "ABC Inc.",
"Location": "Antarctica"
}
```

Array: a pair of square bracket tokens surrounding zero or more values. The values are separated by commas. The order of the values is significant. Here are some examples: 

```json
[1, 2, 3, 4, 5]
[True, False, Null, True]
```

String: a sequence of Unicode code points wrapped with quotation marks. Single quotation mark(") or single backward slash(\) is not allowed in a string with other special characters(see page 10 of [The JSON Data Interchange Standard] [2]). These special characters should be represented by a single backward slash + special character.

<u>Reference:</u>  
----------------

(1). Introducing JSON, _[http://www.json.org/](http://www.json.org/)_. 


[1]: http://www.json.org/        "JSON"
[2]: http://www.ecma-international.org/publications/files/ECMA-ST/ECMA-404.pdf "The JSON Data Interchange Standard"
