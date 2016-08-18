---
layout: post
section-type: post
published: true
category: TECH
tags: [ 'json', 'javascript', 'jquery' ]
title: 'Understand JSON - Part 2: Parse JSON with JavaScript'
date: 2016-02-08
modifiedOn: 2016-03-09
---
[JSON] [1] is widely used in web application by reading data from a web server and display in a web page. With the popularity of web application and the amount of data, [JSON] [1] is also used to transport and parse big files. Besides its benefit as data format, [JSON] [1] files also come as a massive concatenation of characters (one huge string) with encoded strings or arrays inside it which could not be easily consumed by other programming languages, even including JavaScript itself. Therefore, the job of parser which takes this huge string and break it up into data structure comes in place in order to let other programming language work with it smoothly. (Note: Encoding [JSON] [1] is the opposite of parsing [JSON] [1].)

Since [JSON] [1] was derived from JavaScript and its syntax is a subset of the language, it is more natural to use JavaScript to parse [JSON] [1] data. Typicall, there are three different ways to do this:  
1. Use eval() function.  
2. Use JSON.parse() function.  
3. Use JavaScript library, such as jQuery.js.  

__1. Eval()__  
-------------
A quick but dirty way to parse [JSON] [1] data format is to use eval() function. It can be used to convert the well formed JavaScript Object Notation([JSON] [1]) string into an object.   

Syntax: eval(string)

For example:  

```javascript
// Use eval() function to parse JSON

//Example 1:
var myName1 = eval("[ { firstName: 'Yi' }, { lastName: 'Du' } ]");

console.log(myName1[0].firstName);//Yi
console.log(myName1[1].lastName);//Du

//Example 2:
myName2 = eval('(' + '{"firstName": "Yi", "lastName": "Du"}' + ')');

console.log(myName2.firstName);//Yi
console.log(myName2.lastName);//Du

//Example 3:
var newName =  "var myName3 = {firstName: 'Yi', lastName: 'Du'};";

eval(newName);

console.log(myName3.firstName);//Yi
console.log(myName3.lastName);//Du
```

In the two examples above, I first use [JSON] [1] array as string and let it parse through eval() function. The result is persisted in the variable myName1 and could be derived as JavaScript objects by calling each component from the myName1 variable.

However, in the example 2, string used as parameters by eval() function is [JSON] [1] object. Since curly braces could also be used to start a block, JavaScript is confused if we use [JSON] [1] object as parameter to let eval() parse. Therefore, we use parentheses around the expression.

Alternatively, example 3 shows another way to make it clear to eval() by assigning object to a variable within the eval() string.

However, using eval() function is not the best way, actually not even a better way, to parse a [JSON] [1] file for various reasons: 1). security: since eval() can execute any JavaScript program regardless the source, it leaves the door open to a malicious third party to run malicious codes on user's machine with the permission of your website, 2). speed: eval() is also slower than other alternatives since it has to invoke JavaScript interpreter. It is actually a hungry memory eater while executing.
  
  
__2. JSON.parse()__  
---------------
[ECMA5](http://www.ecma-international.org/publications/files/ECMA-ST/Ecma-262.pdf) specifies native support to [JSON] [1]. There are two primary methods for handling [JSON] [1]: JSON.parse (which converts a JSON string into a JavaScript object) and JSON.stringify (which convert a JavaScript object into a serialized string).  

Syntax: JSON.parse(text[, reviver])    

For example:  

```javascript
// Use JSON.parse() function to parse JSON

//Example 4:
myName4 = JSON.parse('{"firstName": "Yi", "lastName": "Du"}');

console.log(myName4.firstName);//Yi
console.log(myName4.lastName);//Du

//Example 5:
myName5 = JSON.parse('{"firstName": 'Yi', "lastName": 'Du'}');

console.log(myName5.firstName);//Uncaught SyntaxError: missing ) after argument list (line 8)
console.log(myName5.lastName);//Uncaught SyntaxError: missing ) after argument list (line 8)
```  

In example 4, JSON.parse() interprets a string and returns its values by calling a variable assigned. However, in example 5, JSON.parse() has difficulty in parsing(almost!) the same string. Remember, the syntax of JSON is only a subset of JavaScript and looking up [JSON] [1] syntax [specification][1], this string _{"firstName": 'Yi', "lastName": 'Du'}_ is not a valid [JSON] [1] since single quotation mark is not JSON-compliant. In this scenario, in order to let JSON.parse() to work properly, only string in example 4 is valid.
  
The _reviver_ parameter is the optional function when calling JSON.parse() function. It takes two parameters: _key_ and _value_ which could filter and transform the results. 

```javascript
//Example 6:
var info = JSON.parse('{"name": "Yi Du", "age": 56}', function(k, v) {
  if (k === '') { return v; } // if topmost value, return it,
  return v / 2;               // else return v / 2.
  
});      
console.log(info.name);//NaN
console.log(info.age);//28
```   

The _reviver_ is ultimately called with the empty string and the topmost value to permit transformation of the topmost value. Be certain to handle this case properly, usually by returning the provided value, or JSON.parse() will return undefined.

JSON.parse() is not supported by all browsers, especially in old browsers. All browsers supporting JSON.parse() could be found [here](http://caniuse.com/#feat=json).

__3. jQuery.js__  
----------------
Both eval() and JSON.parse() function have their own limitations. Sometimes, using JavaScript libraries. There are many libraries that could do this and I will focus only jQuery.js here only.

At the very high level, jQuery.js will use native JSON.parse() method if available. Otherwise, it will try to use its new function which is quite similar as eval() to parse the data.

<u>Reference:</u>  
-----------------

(1). Eval() by MDN, _[https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/eval](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/eval)_.      

(2). Why Won’t eval() Eval My JSON? (Or: JSON Object !== Object Literal), _[http://rayfd.me/2007/03/28/why-wont-eval-eval-my-json-or-json-object-object-literal/](http://rayfd.me/2007/03/28/why-wont-eval-eval-my-json-or-json-object-object-literal/)_.       

(3). ECMAScript 2015 Language Specification, _[http://www.ecma-international.org/ecma-262/6.0/#sec-json.parse](http://www.ecma-international.org/ecma-262/6.0/#sec-json.parse)_.     
 
(4). JSON.pars() by MDN, _[https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON/parse](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON/parse)_.


[1]: http://www.json.org/        "JSON"