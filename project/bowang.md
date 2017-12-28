---
layout: project
section-type: project
title: Project
sitemap:
  priority: 1.0
---

# <a href="http://shiny.rstudio.com/">Shiny Apps</a> Project by <a href="https://www.rstudio.com/">RStudio</a> for <a href="https://yidu.shinyapps.io/bowang">_Bowang Du_</a>
<iframe src="https://ghbtns.com/github-btn.html?user=abrahamdu&repo=bowang&type=star&count=true&size=large" frameborder="0" scrolling="0" width="160px" height="30px"></iframe> 
<iframe src="https://ghbtns.com/github-btn.html?user=abrahamdu&repo=bowang&type=fork&count=true&size=large" frameborder="0" scrolling="0" width="160px" height="30px"></iframe>

When we welcomed our first-born [Bowang](https://yidu.shinyapps.io/bowang) May of 2015, I was thinking about putting all stuff related to him together into a website. Instead of spending money in hosting and server, I realized the efforts by [RStudio](https://www.rstudio.com/) of [Shiny](http://shiny.rstudio.com/) without paying anything so that I can save for infant formula. Please check my [project of Bowang Du](https://github.com/abrahamdu/bowang) in GitHub.  

Here is the R code:  

```r
#### server.R ####

library(shinydashboard)
library(leaflet)
library(sp)
library(maps)

server <- function(input, output) {
    
  output$tabset1Selected <- renderText({
    input$tabset1
  })
 
}

#### ui.R ####

library(shinydashboard)
library(leaflet)
library(sp)
library(maps)
library(RgoogleMaps)
library(RColorBrewer)
library(jpeg)

header <- dashboardHeader(title="Bowang Du")

sidebar <- dashboardSidebar(
             sidebarMenu(
                         menuItem("Announcement",tabName="announcement",icon=icon("bullhorn"),badgeLabel="New Born!",badgeColor="green"),
                         menuItem("Photo",tabName = "photo",icon = icon("fa fa-camera"))
                        )
                           )

body <- dashboardBody(
  #includeCSS("www/bootstrap.css"),
         tags$head(
                   tags$link(rel="stylesheet",type="text/css",href="custom.css")
                  ),
    
         tabItems(
                  tabItem(tabName="announcement",
                   fluidRow(
                    valueBox(
                             "Bowang Du","Name",icon=icon("fa fa-paw"),color="olive"
                            ),
                    valueBox(
                             "May 15, 2015","Birth Date",icon=icon("fa fa-calendar"),color="teal"
                            ),
                    valueBox(
                             "7 lb 2 oz","Weight",icon=icon("fa fa-calculator"),color="red"
                            ),
                    valueBox(
                             "Baby Boy","Gender",icon=icon("fa fa-male"),color="blue"
                            ),
                    valueBox(
                             "11:05:20 AM","Birth Time",icon=icon("fa fa-clock-o"),color="yellow"
                            ),
                    valueBox("19.5 in","Length",icon=icon("fa fa-pencil"),color="purple"
                            ),
                    box(width=12,
                    leaflet() %>% 
                    addTiles() %>%
                    setView(-77.227331,38.8577448, zoom = 16) %>%
                    addMarkers(-77.227331,38.8577448,icon=icons(iconUrl='www/baby-boy-icon.png',iconWidth=60,iconHeight=60)) %>%
                    addPopups(-77.227331,38.8579999,'I was born in <b>INOVA Fairfax</b>!')
                        )  )
                          ),
                   tabItem(tabName = "photo",
                     fluidRow(
                       box(title="First Trio",solidHeader=TRUE,
                         img(src='trio.JPG',width=500,align="center")
                          ),
                       box(title="1 Day",solidHeader=TRUE,
                           img(src='1day.JPG',width=500,align="center")
                       ),
                       box(title="1 Month",solidHeader=TRUE,
                           img(src='1month.JPG',width=500,align="center")
                       ),
                       box(title="2 Month",solidHeader=TRUE,
                         img(src='2month.JPG',width=500,align="center")
                       )
                             )
                          )
               ) 
                  )

  
ui <- dashboardPage(header,sidebar,body,skin = "blue")
```  
