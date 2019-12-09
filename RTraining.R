## R Training file from http://tryr.codeschool.com


#####################################################################
######################## Section 1 ##################################
#####################################################################

# R evaluates what you type into the prompt and prints the answer

#################
## Expressions ##
#################

# Simple math
1+1 #prints 2 onto the console right after the entry

# Strings
"Arr, matey"

# Operations
6*7 #42

####################
## Logical Values ##
####################

# Return TRUE or FALSE values (aka boolean values)

3<4 #Returns TRUE to the screen

2+2==5  #Returns FALSE

#NOTE: Can substitute T for TRUE and F for FALSE as shorthand

T==TRUE

###############
## Variables ##
###############

x <- 42
x/2 #Returns 21

#Can reassign any value to a variable at any time
x <- "Arr, matey"
x

#Can also assign logical values to variables
x <- TRUE

###############
## Functions ##
###############

sum(1,3,5)  #Function name followed by one or more arguments

rep('Yo ho!', times=3)  #Some arguments have names (i.e. times)

sqrt(16)

#################
## Help #########
#################

# help(functionname) brings up help for the given function
help(sum)

help(rep)

# example(functionname) brings up examples of usage for the given function (if applicable)
example(min)

#################
## Files ########
#################

# Can run plain text files written in R script directly from the 
# command line or from within a running R instance

# List files in current directory from within R
list.files()

# NOTE: getwd() prints the current R directory
getwd()

# To run a script, type source(filesource/filename)
source('C:/Users/davidryan/Documents/DTR_Files/RExamples/HelloWorld.R')




#####################################################################
######################## Section 2 ##################################
#####################################################################


#################
## Vectors ######
#################

# Vector is just a list of values
# Values can be numbers, strings, logical values or any other type
# NOTE: Vector values must be of the SAME TYPE!

c(4,7,9)  #NOTE: c() function is short for 'combine'

c('a','b','c')

# NOTE: If you try to put in values of different modes/types, R will sometimes convert entries into something similar
# In this instance, all values are converted into characters so that the vector can hold them all (str() shows that they are characters (chr))
c(1,TRUE,'three')
str(c(1,TRUE,'three'))

c(2, 4, 5, 'a', 'pear', FALSE, TRUE)
str(c(2, 4, 5, 'a', 'pear', FALSE, TRUE))


######################
## Sequence Vectors ##
######################

# Creating a sequence of numbers with start:end notation
5:9

seq(5,9)

#seq allows you to customize increments
seq(5, 9, 0.5)

#You can also create sequences that increment in the reverse direction
9:5

###################
## Vector Access ##
###################

#You can retrieve individual values within a vector by providing numeric index in square brackets
#NOTE: R vector indices start at 1 and not 0

sentence <- c('walk', 'the', 'plank')
sentence[3]   #returns 'plank'

sentence[1]   # returns 'walk'

#Assign new values within existing vector
sentence[3] <- 'dog'

#Add new values to the end; vectors grow to accomodate
sentence[4] <- 'to'

#Use vector within square brackets to access mutliple values
#EX: Retrieve the 1st and 3rd words in the vector
sentence[c(1,3)]  #Returns 'walk' and 'dog'

#Or retrieve ranges of values
sentence[2:4]  #'the' 'dog' 'to'

#You can also set/reset ranges of values
sentence[5:7] <- c('the', 'poop', 'deck')
sencente[6]  #Returns 'poop'


##################
## Vector Names ##
##################

ranks <- 1:3

#Assign names to vector's elements with vector filled with names passed to the names() assignment function
names(ranks) <- c('first', 'second', 'third')
ranks

#Can use names to access the vector's values with ranks[name]
ranks['first']
#Can set values of vector (ranks) to something different by calling the name instead of the position
ranks['third'] <- 100


#########################
## Plotting One Vector ##
#########################

#barplot draws a bar chart when given a vector of values

vesselsSunk <- c(4,5,1)
barplot(vesselsSunk)

#Assigning names to vector values allows names to be used as plot labels
names(vesselsSunk) <- c('England', 'France', 'Norway')
barplot(vesselsSunk)

#Barplot of vector of integers ranging from 1 to 100
barplot(1:100)

#Barplot of sample of 100 values from 0 to 100
barplot(sample(0:100,100,replace=T))


#################
## Vector Math ##
#################

#Most arithmetic operations work just as well on vectors as they do on single values

#Adding a scalar to a vector
a <- c(1, 2, 3)
a+1   #Returns 2 3 4

#Division, multiplication, etc also perform the desired operation on each value in the vector
a/2   #Returns 0.5 1 1.5

a*2   #Returns 2 4 6

#Can also add vectors together
b <- c(4, 5, 6)
a + b   #Returns 5 7 9

a - b   #Returns -3 -3 -3

#Can compare vectors; checks to see which values are equal when sharing indices
a == c(1, 99, 3)
a < c(1, 99, 3)

#Functions can also operate on each element of a vector
sin(a)

sqrt(a)


###################
## Scatter Plots ##
###################

x <- seq(1, 20, 0.1)
y <- sin(x)

plot(x,y) #Plots first argument (x) on the horizontal axis and values from the second argument (y) on the vertical


values <- -10:10
absolutes <- abs(values)

#Plot values on the horizontal axis and absolutes on the vertical axi
plot(values, absolutes)


###################
## NA Values ######
###################

a <- c(1,3,NA,7,9)
sum(a)  #Sum considered 'not available' because one of the vector's values was NA

#Can tell sum (and many other functions) to remove NA before calculating

sum(a, na.rm=TRUE)  #20




#####################################################################
######################## Matrices ###################################
#####################################################################

# Matrices contain information in the form of rows and columns (2-dimensional array)

##############
## Matrices ##
##############

#Create matrix with 3 rows and 4 columns with all fields set to 0
matrix(0, 3, 4)

a <- 1:12
print(a)
#Creates the matrix by filling in down columns then across
matrix(a, 3, 4)

#Can also reshape a vector into a matrix
plank <- 1:8
dim(plank)
dim(plank) <- c(2, 4)
print(plank)  #yields a 2 row x 4 col matrix


###################
## Matrix Access ##
###################

print(plank)

#Provide 2 indices to retrieve values from matrices

plank[2,3]  #6
plank[1,4]  #7

#Can use indices to reassign values
plank[1,4] <- 0


#Obtain an entire matrix row
plank[2,]   #2 4 6 8
#Obtain an entire matrix column
plank[,4]   #7 8
#Reading multiple rows/columns with a vector or sequence with desired indices
#Obtain an entire matrix row
plank[,2:4]   


#####################
## Matrix Plotting ##
#####################

#R has visualations for matrix data so that we can see what is happening when matrices are too large to analyze via text output

elevation <- matrix(1, 10, 10)
elevation[4,6] <- 0

#Contour map of the values
contour(elevation)

#3D Perspective plot
persp(elevation)
persp(elevation, expand=0.2) #Change expand parameter to adjust view so that the highest value is not at the very top

#Use R's built in data sets to explore visualizaitons more
contour(volcano)
persp(volcano, expand=0.2)
#Create a heatmap of the data
image(volcano)




#####################################################################
######################## Section 3 ##################################
#####################################################################

#Tools R has to provide Summary Statistics taht allow you to explain data adequately

#################
## Mean #########
#################

limbs <- c(4, 3, 4, 3, 2, 4, 4, 4)
names(limbs) <- c('One-Eye', 'Peg-Leg', 'Smitty', 'Hook', 'Scooter', 'Dan', 'Mikey', 'Blackbeard')

mean(limbs)

barplot(limbs)

#Draw a line on the plot to represent the mean (allows us to easily compare various values to average)
abline(h=mean(limbs))


#################
## Median #######
#################

limbs[8] <- 14
names(limbs[8]) <- 'Davy Jones'

mean(limbs)

barplot(limbs)
abline(h=mean(limbs))   #Note: Mean is accurate, but also misleading; use 'median' to sort values and choose middle one

median(limbs)

abline(h=median(limbs))   #Plot median line on same graph


########################
## Standard Deviation ##
########################

pounds <- c(45000, 50000, 35000, 40000, 35000, 45000, 10000, 15000)
barplot(pounds)
meanValue <- mean(pounds)
abline(h=meanValue)

# Use standard deviation from the mean to describe the range of typical or normal values for a data set
# For a group of numbers, stdev shows how much they vary from the average value

deviation <- sd(pounds)
abline(h=meanValue+deviation)
abline(h=meanValue-deviation)
#Resulting lines show that the latest two days may be below normal




#####################################################################
######################## Factors ####################################
#####################################################################

# Factors can track categorized values

########################
## Creating Factors ####
########################

chests <- c('gold', 'silver', 'gems', 'gold', 'gems')  #List of strings
types <- factor(chests)   #List of integer reference to one of the 'types' factor levels

print(chests)   #5 strings
print(types)  #5 integer references with 3 levels seen below

as.integer(types)

levels(types)


########################
## Plots With Factors ##
########################

#Use factor to separate plots into categories

weights <- c(300, 200, 100, 250, 150)
prices <- c(9000, 5000, 12000, 7500, 18000)

#Graphing chests by weight and value
plot(weights, prices)

#Use different plot characters for each type by converting the factors into integers and passing it to the pch argument of plot
plot(weights, prices, pch=as.integer(types))

#Add a legend to show what symbols mean
legend('topright', c('gems','gold','silver'),pch=1:3)
#Derive labels and plot characters instead to avoid updating hardcoding
legend('topright', levels(types),pch=1:length(levels(types)))




#####################################################################
######################## Data Frames ################################
#####################################################################

# Tying data structures together into a single entity
# Much like a database table or Excel spreadsheet
# Certain number of columns, each of which contains values of a particular type
# Indeterminate number of rows (set of related values for each column)

#################
## Data Frames ##
#################

#Convert weights, prices and types variables into a dataframe
treasure <- data.frame(weights, prices, types)
print(treasure)


########################
## Data Frame Access ###
########################

# Get individual columns in a dataframe by providing their index number in double brackets
treasure[[2]]
# Get individual columns in a dataframe by providing their index name in double brackets
treasure[['weights']]
# Shorthand call notation to avoid numerous bracket use (use $)
treasure$prices


##########################
## Loading Data Frames ###
##########################

list.files()

#Load CSV file contents with read.csv()
read.csv('C:/Users/davidryan/Documents/DTR_Files/TrainingMaterials/targets.csv')

#Load tab delimited data with read.table()
read.table('C:/Users/davidryan/Documents/DTR_Files/TrainingMaterials/infantry.txt', sep='\t', header=T)


##########################
## Merging Data Frames ###
##########################

targets <- read.csv('C:/Users/davidryan/Documents/DTR_Files/TrainingMaterials/targets.csv')
infantry <- read.table('C:/Users/davidryan/Documents/DTR_Files/TrainingMaterials/infantry.txt', sep='\t', header=T)

# Use the contents of one or more columns to merge two data frames together
merge(x=targets, y=infantry)

#Select only those columsn of interest
mergeF <- merge(x=targets, y=infantry)
mergeF <- mergeF[c(1,3:5)]




#####################################################################
######################## Real-World Data ############################
#####################################################################

#File with piracy rate, sorted by country [ piracy ]
#File with GDP per capita for each country (wealth produced/population) [ gdp ]
gdp <- read.csv('C:/Users/davidryan/Documents/DTR_Files/TrainingMaterials/gdp.csv')
str(gdp)
gdp$Country <- as.character(gdp$Country)

piracy <- read.csv('C:/Users/davidryan/Documents/DTR_Files/TrainingMaterials/piracy.csv')
str(piracy)
piracy$Country <- as.character(piracy$Country)

piracyNACountries <- piracy[is.na(piracy$Piracy),]
piracyNACountries <- piracyNACountries$Country
piracy <- na.omit(piracy)

#Merge these on the country names countries = merge(x=gdp, y=piracy)
countries = merge(x=gdp, y=piracy)

#Plot Piracy (y) vs. GDP (x) [ plot(countries$GDP, countries$Piracy) ] - shows a negative correlation (higher a countries GDP, the lower the % of pirated software)
plot(countries$GDP, countries$Piracy)

#Take a look at lower GDP countries to see if there is a trend
countries2 <- countries[countries$GDP<5000000,]
plot(countries2$GDP, countries2$Piracy)

#Test for correlation between two vectors with the cor.test() function 
cor.test (countries2$GDP, countries2$Piracy)
#Look at p-value - if it is below 0.05, then there likely is a correlation (i.e. there is a statistically significant negative correlation between GDP and software piracy)

#Since we have more countries in our GDP data than in our piracy data, can we use a known GDP to estimate a piracy rate?
# A: calculate the linear model that best represents all data points
# A: lm() takes a model fomula represented by a response variable, a tilde and a predictor variable (GDP)
y <- countries2$Piracy
x <- countries2$GDP
line2 <- lm(y ~ x)
abline(line2)

summary(line2)

opar <- par(mfrow = c(2,2), oma = c(0, 0, 1.1, 0))
plot(line2, las = 1)
par(mfrow = c(1,1))

missingCountries <- gdp[gdp$Country %in% piracyNACountries,]
    
missingCountries$Piracy <- predict(line2, data.frame(x=missingCountries$GDP))

print(missingCountries)

plot(missingCountries$GDP, missingCountries$Piracy, xlab='GDP', ylab='Piracy')


##############
## ggplot2 ###
##############

#One of many functionalities included as a library on the servers of CRAN (Comprehensive R Archive Network) - statistical functions to graphics

#Popular graphics package
install.packages('ggplot2')
library("ggplot2", lib.loc="~/R/win-library/3.0")

help(package='ggplot2')

plot(weights, prices)
#Use qplot - a commonly used part of ggplot2 to create a plot with a background, legend and colored points without hassels of configuration in plot()
qplot(weights, prices, color=types)
qplot(weights, prices, color=types, size=1)
qplot(weights, prices, color=types, size=types)

