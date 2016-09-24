---
layout: post
section-type: post
published: true
category: TECH
tags: [ 'jekyll', 'web', 'css' ]
title: 'Syntax Highlighter of Jekyll with Rouge'
date: 2016-09-22
modifiedOn: 2016-09-22
---

On Feb 1, 2016, GitHub Pages announced to only support [Rouge] [2] for syntax highlighting (see [announcement](https://github.com/blog/2100-github-pages-now-faster-and-simpler-with-jekyll-3-0)). This means you don't have to install __Python__ in order for *Pygments* to highlight syntax. 

[Rouge] [2] is a pure __Ruby__ syntax highlighter. It can highlight more than [100 languages](https://github.com/jneen/rouge/wiki/List-of-supported-languages-and-lexers) and use [backtick-style fenced code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks/) within the [kramdown](http://kramdown.gettalong.org/) markdown which is only supported by GitHub Pages starting from May 1st, 2016 in the same [announcement](https://github.com/blog/2100-github-pages-now-faster-and-simpler-with-jekyll-3-0).   

Before using [Rouge] [2], you will need to install it first:  

```ruby
## Install Rouge
gem install rouge
## Fetching: rouge-2.0.6.gem (100%)
## Successfully installed rouge-2.0.6
## Parsing documentation for rouge-2.0.6
## Installing ri documentation for rouge-2.0.6
## Done installing documentation for rouge after 22 seconds
## 1 gem installed
```  

Under [Jekyll] [1], you will also need to change *_config.yml* file:  

```ruby  
markdown: kramdown
kramdown:
  input: GFM
  syntax_highlighter: rouge
```  

According to [documentation](https://github.com/jneen/rouge#full-options), there are three ways of customizing the themes. I personally only use *CSS Theme options* and will go through how to set it up.  

Command _rougify_ ships with [Rouge] [2] and we can use it to first check available CSS themes for highlighting:  

```ruby  
rougify help style
## usage: rougify style [<theme-name>] [<options>]
## 
## Print CSS styles for the given theme.  Extra options are
## passed to the theme.  Theme defaults to thankful_eyes.
##
## options:
##   --scope       (default: .highlight) a css selector to scope by
##
## available themes:
##   base16, base16.dark, base16.monokai, base16.monokai.light, base16.solarized, ##   base16.solarized.dark, colorful, github, gruvbox, gruvbox.light, molokai, 
##   monokai, monokai.sublime, thankful_eyes, tulip
```  

In the end, we can find available themes for CSS to use. I pick up *base16.solarized.dark* theme to generate CSS file for my [test site](http://anotherpeak.org/test/) syntax highlighting.  

```ruby
rougify style base16.solarized.dark > css/syntax.css
```  

And I include this generated CSS file into markdown file:  

```html
<link href="/test/css/syntax.css" rel="stylesheet">
```

Now we can see the corresponding languages are highlighted by [Rouge] [2] [here](http://anotherpeak.org/test/jekyll/update/2016/08/27/welcome-to-jekyll.html).  


[1]: https://jekyllrb.com/  "Jekyll"
[2]: http://rouge.jneen.net/  "Rouge"