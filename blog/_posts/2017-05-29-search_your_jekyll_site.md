---
layout: post
section-type: post
published: true
category: TECH
tags: [ 'jekyll', 'web', 'css', 'javascript' ]
title: 'Search Your Jekyll Site'
date: 2017-05-29
modifiedOn: 2017-05-29
---

When I started to use [Panos Sakkos](https://panossakkos.github.io/)'s _[Jekyll Personal Theme](http://jekyllthemes.org/themes/personal/)_ to run my own [Jekyll] [1] site, I find the template is missing a site searching function. After a couple of months blogging and other things added into the site, it becomes hard to find a specific content without a search option. 

I start to do research by myself to see what's the best option to install a search function into my [own website](http://anotherpeak.org/). Eventually it turns out that I have three options: 1). Google Custom Search Engine; 2). [Lunr.js](https://lunrjs.com/); 3). [Simple Jekyll Search](https://christianfei.com/posts/Use-Simple-Jekyll-Search-on-your-blog-in-these-easy-steps/) by Christian Fei. The first option is definitely the easiest to implement and light but not instant yet. The second is instant search but heavy. [Katy DeCorah](http://katydecorah.com/) blogged to iterate how to include [Lunr.js](https://lunrjs.com/) into [Jekyll] [1] site and her final search implementation is quite neat (Please check [this link](http://katydecorah.com/search/)). The third option is actually both instant and light.

After doing all these researches with an idea of how I would like my own search function works out in my site, I set the following rules to narrow down my choices:  
<li>1. Instant and Light</li>  
<li>2. Full text client-side search engine</li>  
<li>3. Popular web search engine alike</li>  

These rules actually put myself into the spot to adopt [Simple Jekyll Search](https://christianfei.com/posts/Use-Simple-Jekyll-Search-on-your-blog-in-these-easy-steps/) as my selection. And I just need to read the documents of it and put my hands on to implement it. However, after long-search on the web, I suddely find [Ben Howdle](http://benhowdle.im/)'s [2014 blog](http://benhowdle.im/creating-a-dynamic-search-page-for-your-jekyll-blog.html) and [Mat Hayward](http://mathayward.com/)'s [2014 blog](http://mathayward.com/jekyll-search/) which both enlighten me. __This is the search option, I want!__   

__1. Add Search Option into Navigation__  
----------------------------------------    
Since my current site's navigation didn't have search tab, I need add it into the list of navigation. Here is what I come out:  

```javascript  
<!--  SEARCH -->
          <li class="li-form">
              <a>
              <form role="search" id="search-nav" method="GET" action="{{ '/search' | prepend: site.baseurl }}">
                 <input type="search" class="search-field" placeholder="What are you looking for?" value="" name="query" title="Rechercher :" style="color:black;background-color:white;">
                  <button type="reset">
                        <span class="fa fa-close" style="color:black;">
                          <span class="sr-only">Close</span>
                        </span>
                    </button>
                    <button type="submit" class="search-submit">
                        <span class="fa fa-search" style="color:gold;">
                          <span class="sr-only">Rechercher</span>
                        </span>
                    </button>
              </form>
              </a>
          </li>
```  

This creates the search tab into my navigation list and allows user to input whatever they want search my site.  

__2. Add Search Layout, Index Page and JS Search Function__  
-----------------------------------------------------------  

I also create 'search' layout using 'blog' and create _search.html_ index file under my home directory. This index file is essentially a mixture of Ben and Mat's index file for search.  
  
```javascript   
//Mat's  
<p data-search-found>
    <span data-search-found-count></span> result(s) found for &ldquo;<span data-search-found-term></span>&rdquo;.
</p>

//Ben's  
<div id="search_results">
</div>

<script>
	var JEKYLL_POSTS = [];
	...
</script>
<script src="{{ "/js/search.js" | prepend: site.baseurl }}"></script>
<script>
	new jekyllSearch({
		selector: "#search_results",
		properties: ["title", "content"]
	});
</script>
```  

You may notice in Ben's code, it calls a _search.js_ under _js_ folder. This is also basically from Ben's _[search.js](https://github.com/benhowdle89/jekyll-search-demo/blob/gh-pages/js/search.js)_ and I just reference Mat's code from his _[search.js](https://github.com/mathaywarduk/jekyll-search/blob/master/scripts/search.js)_ to add the count of returned results and search term from user.  

```javascript  
var $foundCount = $("[data-search-found-count]"),
        $foundTerm = $("[data-search-found-term]");
    function populateResultsCount(count) {
        $foundCount.text(count);
    }
    function populateResultsString(term) {
        $foundTerm.text(term);
    }

var scanPosts = function(posts, properties, query) {
		var results = [],resultsCount = 0,resultsTerm = query;
		posts.forEach(function(post) {
			var textToScan = "",
				regex = new RegExp(query, "ig");
            
			properties.forEach(function(property) {
				if (post.hasOwnProperty(property)) {
					textToScan += post[property];
				}
			});
            
			if (regex.test(textToScan)) {
				results.push(post);
                resultsCount++;
			}
		});
        console.log(populateResultsCount(resultsCount));
        console.log(populateResultsString(resultsTerm));
		return results;
        return populateResultsCount(resultsCount);
        return populateResultsString(resultsTerm);
	};
```  

__3. Configuration__  
--------------------  

With all these steps ready, I just need to add some cosmetics stuff into my site file.  

```javascript  
\\Search Page Image from _config.yml
search-img: "/img/search.jpg"  

\\Search button from grayscale.scss
#search-nav input, #search-nav button[type="reset"] {
  display: none; }

#search-nav input, #search-nav button {
  border-radius: 0px;
  border-width: 0px;
  color: #454545;
  background-color: transparent;
  border: none;
  box-shadow: none;
  outline: none; }

#search-nav .search-submit {
  position: relative;
  bottom: 2px; }
```  

Eventually, I have my own search page with all three criteria satisfied. Although it is still not the ideal one I'd like to see, it fits into my needs and expectations quite well. Please check if you feel interested in: [http://anotherpeak.org/search](http://anotherpeak.org/search).    

For my following work for my search engine, I'd expand it to search the whole site instead of just blogs. Also, I hope it would consume mutiple key words. Stay tuned and I would share once I have these tasks accomplished!  

<br />

<u>Reference:</u>    
-----------------  
(1). Use Simple-Jekyll-Search on your blog in these easy steps, _[https://christianfei.com/posts/Use-Simple-Jekyll-Search-on-your-blog-in-these-easy-steps/](https://christianfei.com/posts/Use-Simple-Jekyll-Search-on-your-blog-in-these-easy-steps/)_.  
(2). Jekyll search with JSON, _[http://mathayward.com/jekyll-search/](http://mathayward.com/jekyll-search/)_.  
(3). Jekyll Simple Search, _[https://blog.webjeda.com/jekyll-search/](https://blog.webjeda.com/jekyll-search/)_.    
(4). Creating a dynamic search page for your Jekyll blog using JavaScript, _[http://benhowdle.im/creating-a-dynamic-search-page-for-your-jekyll-blog.html](http://benhowdle.im/creating-a-dynamic-search-page-for-your-jekyll-blog.html)_.  
(5). Simple Jekyll searching, _[https://alexpearce.me/2012/04/simple-jekyll-searching/](https://alexpearce.me/2012/04/simple-jekyll-searching/)_.  
(6). lunr.js and Jekyll, _[http://katydecorah.com/code/lunr-and-jekyll/](http://katydecorah.com/code/lunr-and-jekyll/)_.  

[1]: https://jekyllrb.com/  "Jekyll"