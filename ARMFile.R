# -*- coding: utf-8 -*-


install.packages("arules")

install.packages("devtools")
library(devtools)

install.packages("TSP")

install.packages("data.table")

install.packages("sp")

install.packages("arulesViz", dependencies = TRUE)

install.packages("datasets.load")

library("devtools")
install_github("mhahsler/arulesViz")

install.packages("dplyr", dependencies = TRUE)
install.packages("purrr", dependencies = TRUE)
install.packages("devtools", dependencies = TRUE)
install.packages("tidyr")

install_github("mhahsler/arulesViz")



Data <- read.csv("D:/Boulder/Machine Learning/Wholedata.csv")

library(arulesViz)

head(Data)

Data <- na.omit(Data)

summary(Data$Spend..USD)

colnames(Data)

bin_data <- function(x, string) {
  quartiles <- quantile(x, probs = c(0, 0.25, 0.5, 0.75, 1))
  ifelse(x <= quartiles[2], paste("Low" , string),
         ifelse(x <= quartiles[3], paste("Moderate " , string),
                ifelse(x <= quartiles[4], paste("High" , string), paste("Very High ", string))))
}

Data$Spend..USD. = bin_data(Data$Spend..USD., "Spending")

Data$X..of.Impressions = bin_data(Data$X..of.Impressions, "Impressions")

Data$Reach = bin_data(Data$Reach, "Reach")

Data$X..of.Website.Clicks = bin_data(Data$X..of.Website.Clicks, "Website Clicks")

Data$X..of.Searches = bin_data(Data$X..of.Searches , "Search")

Data$X..of.View.Content = bin_data(Data$X..of.View.Content , "Viewing")

Data$X..of.Add.to.Cart = bin_data(Data$X..of.Add.to.Cart , "Carting")

Data$X..of.Purchase = bin_data(Data$X..of.Purchase , "Purchasing")

head(Data)

ARMData <- cbind.data.frame(Data[,1], Data[,3:10])


library(arules)
library(arulesViz)

write.csv(ARMData, "D:/Boulder/Machine Learning/ARMData.csv")

library(arules)

ABARM <- read.transactions("D:/Boulder/Machine Learning/ARMData.csv",
                           rm.duplicates = FALSE,
                           format = "basket",  ##if you use "single" also use cols=c(1,2)
                           sep=",",  ## csv file
                           cols=1, skip = 1)

FirstRule = arules::apriori(ABARM, parameter = list(support=0.2,
                                                    minlen=2))

inspect(FirstRule)

SecondRule = arules::apriori(ABARM, parameter = list(confidence =0.9,
                                                     minlen=2))

inspect(SecondRule)

SecondRule[1]




# Load the arulesViz package
library(arulesViz)

SortedRules <- sort(SecondRule, by="confidence", decreasing=TRUE)
inspect(SortedRules[1:10])
(summary(SortedRules))
plot(SortedRules, method="graph", engine="htmlwidget")
plot(SortedRules)
subrules <- head(sort(SortedRules, by="lift"),10)

SortedRules1 <- sort(FirstRule, by="support", decreasing=TRUE)
subrules2 <- head(sort(SortedRules1, by="support"),10)
plot(subrules)

plot(subrules, method="graph", engine="htmlwidget")
title("Network Of Support"  , 
      cex.main = 4, font.main = 3, col.main = "darkgreen", 
      cex.sub = 2, font.sub = 3, col.sub = "darkgreen", 
      col.lab ="black"
) 
plot(subrules2)

plot(subrules2, method="graph", engine="htmlwidget", main="Graph of Support")

