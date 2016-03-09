rm(list = ls())

library(ggplot2)
library(GA)

# read the data and delete "GB_locus" column
# data is avaliable at http://anirbanmukhopadhyay.50webs.com/data.html
data <- read.csv("SpinalCordOriginalData", sep = "")
data$GB_locus <- NULL
data$gene <- NULL
write.table(data, file = "rat_spinal_cord_data.txt", row.names = FALSE, col.names = FALSE)

# fitness <- function(chromosome){
#   data[chrome = 1, ]
# }
# GA <- ga(type = "binary", fitness = fitness, min = -20, max = 20)


# p <- c(6, 5, 8, 9, 6, 7, 3)
# w <- c(2, 3, 6, 7, 5, 9, 4)
# W <- 9
# knapsack <- function(x) {
#   f <- sum(x * p)
#   penalty <- sum(w) * abs(sum(x * w) - W)
#   f - penalty
# }
# GA <- ga(type = "binary", fitness = knapsack, nBits = length(w),
#          maxiter = 1000, run = 200, popSize = 20)
# summary(GA)