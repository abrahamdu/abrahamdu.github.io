---
layout: post
section-type: post
published: true
category: TECH
tags: [ 'markdown', 'markup', 'web', 'html' ]
title: 'Differences between Markup and Markdown'
date: 2016-03-15
modifiedOn: 2016-03-15
---

When my department starts to set up our own wiki web page and ask us to share information around it, I noticed we could create web pages or posts through Markup, besides more user-friendly(to most people in our team without appropriate knowledge of web technique) text editor. After looking at its syntax which is shown right next to web page if you are on edit mode, I feel like it pretty much the Markdown syntax these days I have been studying and digging. For example, it allows users to highlight fonts by some simple syntax. But all of a sudden, I recognized it quite different from Markdown when I tried to edit a hyper link into the web page.

First thing first, the stuff here I am talking about is not a concept in finance/accounting area. Please go to [this link](https://in.answers.yahoo.com/question/index?qid=20110607054830AA5TWHM) to check details. It is funny to see the same terminology used in totally different fields.

However, the Markup and Markdown I am talking about here is the language used in web development. Markup is the programming language which uses markup elements, such as <body>, <div>, etc. to define the layout of pages and also uses other markup elements, such as <em>, <span> to give pages contents. HTML and XML are two widely used flavors of Markup languages in creating web pages.  

HTML Example:

```html  
/*HTML Example*/
<!DOCTYPE html>
<html>
  <head>
    <title>This is a blog</title>
  </head>
  <body>
    <p>Hello world!</p>
  </body>
</html>
```
 
XML Example:  

```xml
/*XML Example*/
<?xml version="1.0" encoding="ISO-8859-1" ?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
</xs:schema>
```

On the other hand, [Markdown](https://daringfireball.net/projects/markdown/) is a lightweight Markup language which allows users to write using (almost!)plain text to convert text to HTML/XHTML by _John Gruber_. According to him, Markdown is not meant to replace Markup, such as, Instead, it is just a very small subset of HTML tags to make users easy to read, write and edit web pages. By addressing this, Markdown is specified to be able to do _block elements_, such as paragraphs and line breaks, code blocks and _span elements_, such as links and images. The full syntax could be found: [http://daringfireball.net/projects/markdown/syntax#overview](http://daringfireball.net/projects/markdown/syntax#overview). On the other hand, if what you need to edit a web page could be covered by Markdown, you could simply use Markup tags to indicate a switch from Markdown to Markup for editing. However, any Markdown syntax couldn't be included into Markup language.

```markdown
This is an H1
=============

## This is an H2

_italic_

__bold__

[link](http://localhost:4000/)

```  

Since _John Gruber_ created Markdown language in 2004, there is no clear definition of it standard other than the initial specifiction. So there are dozens of different flavors of Markdown out there, all with differences in the way they behave. It has been proposed by a couple of developers to standardize the whole language and let major web hosting services, such GitHub, Stack Exchange and Reddit to comply with official Markdown specification rather than having their own tweaked Markdown implementations. 

