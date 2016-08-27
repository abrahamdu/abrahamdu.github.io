---
layout: post
section-type: post
published: true
category: TECH
tags: [ 'jekyll', 'web' ]
title: 'Build Jekyll Website from Scratch on Windows'
date: 2016-08-25
modifiedOn: 2016-08-25
---

When I decided to open my own website with some basic functions, such as blogging, hosting GitHub projects, [Jekyll] [1] comes as a natural choice to serve as a static site generator with GitHub pages. However, the sad thing is that I didn't have any knowledge of web development, hosting that time and my first [Jekyll] [1] effort is basically a fork of GitHub project _[Jekyll Now](https://github.com/barryclark/jekyll-now)_. But working on a project in GitHub directly lacks testing, control and monitoring which pushes me further to dig into [Jekyll] [1].  

__1. Introducing [Jekyll] [1] and [GitHub Pages] [2]__   
-------------------------------------------------------  
Once opening account through GitHub, each account is allowed to have one site hosted directly from GitHub repositories which is called '_[GitHub Pages] [2]_'. Please go to this site ([GitHub Pages] [2]) for details.  

[Jekyll] [1] is a simple, blog-aware, static site generator. It uses static _html_ pages so there is no extra requirement to database which makes the loads quite fast. Also, [Jekyll] [1] is the engine behind [GitHub Pages] [2] which means people could host sites right from GitHub repositories. Combining them together, it gives people ways of creating sites with fast blogging and project sharing with such benefits: 1). local edit; 2). control testing by running [Jekyll] [1]; 3). pushing through Git.

After checking the official website of [Jekyll] [1], I notice that Windows is not official supported (see: [http://jekyllrb.com/docs/windows/](http://jekyllrb.com/docs/windows/)). Plus, I have no idea of _Ruby_. I decide to learn how to set up [Jekyll] [1] on _Windows_ and record the necessary steps based on my experience as follows.  

__2. Install Ruby and the Ruby DevKit__   
-------------------------------------------------------  
Since [Jekyll] [1] is written by Ruby, you will need to install Ruby and corresponding Ruby DevKit which is a toolkit that makes it easy to build and use native C/C++ extensions such as RDiscount and RedCloth for Ruby on Windows.  

(2a). Install Ruby and DevKit 
Go to [this site](http://rubyinstaller.org/downloads/) to install Ruby first. Currently, the latest version of Ruby is _2.3.1_. 

![alt text](/img/blog/ruby1.png)  

After installing Ruby successfully, go ahead to install corresponding DevKit.

![alt text](/img/blog/rubydevkit1.png)  

(2b). Bind DevKit to Ruby on Windows  
Then I follow the steps [here](https://github.com/oneclick/rubyinstaller/wiki/Development-Kit) to bind it to Ruby installation in my path.  

![alt text](/img/blog/rubydevkit2.png)  

__3. Install [Jekyll] [1]__   
-------------------------------------------------------  
Installing [Jekyll] [1] is done by the form of a Ruby Gem. To start with, I launch the command line from the Ruby just installed in step 2. First of all, I can check which Gem already exists in the Ruby by:  
 
```ruby  
gem list  
## *** LOCAL GEMS ***
##
## bigdecimal (1.2.8)
## did_you_mean (1.0.0)
## io-console (0.4.5)
## json (1.8.3)
## minitest (5.8.3)
## net-telnet (0.1.1)
## power_assert (0.2.6)
## psych (2.0.17)
## rake (10.4.2)
## rdoc (4.2.1)
## test-unit (3.1.5)
```  
  
To install [Jekyll] [1] and its dependencies, enter the following command and wait a few seconds until it is completed finished:  

```ruby  
gem install jekyll

gem list
## *** LOCAL GEMS ***
## 
## bigdecimal (1.2.8)
## colorator (1.1.0)
## did_you_mean (1.0.0)
## ffi (1.9.14 x86-mingw32)
## forwardable-extended (2.6.0)
## io-console (0.4.5)
## jekyll (3.2.1)
## jekyll-sass-converter (1.4.0)
## jekyll-watch (1.5.0)
## json (1.8.3)
## kramdown (1.12.0)
## liquid (3.0.6)
## listen (3.0.8)
## mercenary (0.3.6)
## minitest (5.8.3)
## net-telnet (0.1.1)
## pathutil (0.14.0)
## power_assert (0.2.6)
## psych (2.0.17)
## rake (10.4.2)
## rb-fsevent (0.9.7)
## rb-inotify (0.9.7)
## rdoc (4.2.1)
## rouge (1.11.1)
## safe_yaml (1.0.4)
## sass (3.4.22)
## test-unit (3.1.5)
```  

You will see [Jekyll] [1] has been installed into my local machine with its dependencies, such as _kramdown_ and _rouge_. 

Since I will create my site with [Jekyll] [1] as the engine and host it on [GitHub Pages] [2], it is good to know which versions of [Jekyll] [1] and its dependencies are supported by [GitHub Pages] [2]. Here is the [link](https://pages.github.com/versions/). 

After all the steps above are successfully completed, we can create a simple [Jekyll] [1] by running the following command:  

```ruby  
jekyll new test
## New jekyll site installed in C:/Users/abrah/Documents/GitHub/test.
```

A new folder under current directory is created with the following basic files/folders:  

![alt text](/img/blog/jekyll1.png)  

With these minimal(default) files and folders, we finish building our first [Jekyll] [1] site. In order to see this testing site in our local machine, we still need to install the following Gems:  

```ruby
gem install wdm
gem install bundler
bundle install

bundle exec jekyll serve 
##run Jekyll through Ruby
```  

[Jekyll] [1] has a built-in feature to allow us to keep track of changes. In Windows, you need to install Gem 'wdm' to enable this function. Also, suggested by [GitHub Pages] [2], Ruby uses the contents of your Gemfile to track site's dependencies and versions. Here is the content of Gemfile of this test site:  

```ruby  
source "https://rubygems.org"
ruby RUBY_VERSION
gem "jekyll", "3.2.1"
gem "minima"
```  

Now, our first [Jekyll] [1] site is up and running in my local machine:  

![alt text](/img/blog/jekyll2.png)  


<u>Reference:</u>  
-----------------
(1). Jekyll official website, _[https://jekyllrb.com/](https://jekyllrb.com/)_.  
(2). GitHub Pages official website, _[https://pages.github.com/](https://pages.github.com/)_.  
(3). Run Jekyll on Windows, _[http://jekyll-windows.juthilo.com/](http://jekyll-windows.juthilo.com/)_.
(4). Jekyll on Windows, _[https://jekyllrb.com/docs/windows/](https://jekyllrb.com/docs/windows/)_.



[1]: https://jekyllrb.com/  "Jekyll"
[2]: https://pages.github.com/ "GitHub Pages"
