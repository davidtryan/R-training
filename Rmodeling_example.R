






# Getting Data Into R
Data <- read.table("C:/Users/davidryan/My Documents/DTR_Files/TrainingMaterials/R/LogisticRegressionCourse/Data/CH01PR19.txt",header=F)
names(Data) <- c("GPA","ACT")
head(Data)

###########################################################################

# Making a scatterplot of the data
par(mfrow = c(1,1))
#plot(x,y,...)
plot(Data$ACT, Data$GPA, main = "GPA vs. ACT Plot", xlab="ACT Score",
     ylab="Freshman GPA", pch=20)

###########################################################################

# Simple Linear Regression
College <- lm(GPA ~ ACT, data=Data)
summary(College)    

#plot regression line
abline(College$coefficients[1], College$coefficients[2])

#anova significance test
anova(College)

#vector of fitted values
College$fitted.values
#vector of residuals
College$residuals

###########################################################################

## Regression Inferences in R

#95% Confidence intervals for the regression parameters in the linear model
c95 <- confint(College)
abline(c95[1,1],c95[2,1])
abline(c95[1,2],c95[2,2])

#99% Confidence intervals for the regression parameters in the linear model
c99 <- confint(College, level=0.99)
abline(c99[1,1],c99[2,1])
abline(c99[1,2],c99[2,2])

summary(College)    #NOTE: Small ACT p-value (0.00292) and large t-value (3.040) allow you to reject the Ho at any reasonable significance level and
#conclude that there IS a linear association between the two variables

#confidence intervals for the mean response E{Yh} at different specified levels of your explanatory variable
new <- data.frame(ACT=c(20,25,30))
CI <- predict(College, new, se.fit=T, interval="confidence", level= 0.90)
PI <- predict(College, new, se.fit=T, interval="confidence", level= 0.90)

#Working-Hotelling 1-alpha confidence band for the regression line at Xh
W <- sqrt(2*qf(0.90,2,118))
#confidence band at Xh; 90% confidence band when Xh = 25
c(CI$fit[2,1] - W * CI$se.fit[2], CI$fit[2,1] + W * CI$se.fit[2] )


###########################################################################

## More Regression Inferences in R

#anova significance test
anoCollege <- anova(College)    #Use p-value of 0.002917 to reject null hypothesis or compare F* to F to do the same thing (if F* in the anova table > F, then reject Ho)
#Finding the F value
n <- dim(Data)[1]
Fval <- qf(0.95, 1, n-2)   #F(1-alpha, 1, n-2)....0.95 = 1-alpha where alpha=0.05; 118 = n=120
if (anoCollege[[1,"F value"]]>Fval) {sprintf("F*(anova) exceeds F-value(table) - Reject Ho") } else {sprintf("F*(anova) less than F-value(table) - Accept Ho")}

#Pearson product-moment correlation coefficient r12 (which is an estimator of p12) between your predictor and response variables 
cor(Data)
#The two off-diagonal values (which will always be the same) are equivalent to r12, in this case 0.2694818.  You can use this value to calculate t* under equation (2.87)  and conduct a t-test for linear independence of the two variables

#Use the value (0.3127847) in the off-diagonal of the resulting matrix as rs in equation (2.101) to perform the t-test for linear independence of the two variables.  To conduct the test, first capture the correlation:
#the Spearman rank correlation coefficient rs
r <- cor(Data, method="spearman")[1,2]
t <- r*sqrt(n-2)/sqrt(1-r^2)
#Then compare this value to the critical value under the t distribution with n - 2 degrees of freedom
#qt(1-alpha/2,n-p)
Tval <- qt((1-(0.05/2)),n-2)
if (t>Tval) {sprintf("t*(rs) exceeds t-value(table) - Reject Ho: two variables are linearly dependent") } else {sprintf("t*(rs) less than t-value(table) - Accept Ho: variables are linearly independent")}


###########################################################################
## More Regression Inferences in R

#Add model residuals and predicted values to data table
Data <- cbind(Data, College$fitted.values, College$residuals)
names(Data)[3:4] <- c("fitted.values", "residuals")

#stem and leaf plot of the residuals
stem(Data$residuals, scale=2)
#boxplot of the residuals
boxplot(Data$residuals, ylab="residuals", pch=19)
#histogram of residuals
hist(Data$residuals, ylab="residuals")
#index (time sequence) plot of the residuals
plot(Data$residuals, type="p", main="Index Plot", ylab="Residuals", pch=19)
abline(h=0)
#plot residuals against the predictor ACT
plot(Data$ACT, Data$residuals, main="Residuals vs. Predictor", xlab="ACT Score", ylab="Residuals", pch=19)
abline(h=0)
#plot residuals against predicted values
plot(Data$fitted.values, Data$residuals, main="Fitted Values vs. Predictor", xlab="Fitted Values", ylab="Residuals", pch=19)
abline(h=0)

#normal probability plot of the residuals
ptest<-qqnorm(Data$residuals, main="Normal Probability Plot", pch=19)
qqline(Data$residuals)

library(car)
qqPlot(Data$residuals, dist="norm", col=palette()[1],
       ylab="Residual Quantiles", main="Normal Probability Plot", pch=19)
#NOTE: If the relationship between the theoretical percentiles and the sample percentiles is approximately linear, the normal 
#probability plot of the residuals suggests that the error terms are indeed normally distributed.



#To obtain the correlation between the ordered residuals and their expected values under normality, so that you can 
#determine whether the residuals are normally distributed, you will need to carefully follow these steps:
StdErr <- summary(College)$sigma
ExpVals <- sapply(1:n, function(k) StdErr*qnorm((k-0.375)/(n+0.25)))
cor(ExpVals, sort(College$residuals))

#https://78d0cafdec1787a7bff1fbca7d10f12af8369dc8.googledrive.com/host/0Byp6EJD-1552LXRCaFgxWXpJSHM/KELAS_3SE5/ANAREG/E-BOOK/ebook%20regresi/%5BMichael_H_Kutner,_Christopher_J._Nachtsheim,_John(BookFi.org).pdf
#See page 1329 Table B.6
#To conclude at the alpha level that the residuals are normally distributed, the correlation must exceed the corresponding critical value.  
#since the smallest critical value at n = 100 is 0.979 at alpha = 0.005, and the critical values get larger as n gets larger, we conclude that we would not reject the null hypothesis that the residuals are not 
#normally distributed even at the 0.005 level, since our correlation is smaller than 0.979.
#CONCLUSION: RESIDUALS ARE NORMALLY DISTRIBUTED


#Breusch-Pagan test for constancy of error variance (assuming the residuals are independent and normally distributed)
install.packages("lmtest")
library(lmtest)
bptestval <- bptest(College, studentize=F, data=Data)
if (as.numeric(bptest(College, studentize=F, data=Data)$p.value)<0.05) {sprintf("P-value is below a-value - Reject Ho: error variance is not constant") } else {sprintf("P-value is above a-value - Accept Ho: error variance is constant")}

#the Brown-Forsythe Test for constancy of error variance
#Split residuals into two groups
Group1 <- Data[which(Data$ACT < 26), "residuals"]
Group2 <- Data[which(Data$ACT > 25), "residuals"]
#Obtain group medians
M1 <- median(Group1)
M2 <- median(Group2)
#mean absolute deviation for each group
D1 <- sum(abs(Group1-M1))/length(Group1)
D2 <- sum(abs(Group2-M2))/length(Group2)
#pooled standard error
s <- sqrt((sum((abs(Group1-M1)-D1)^2)+sum((abs(Group2-M2)-D2)^2))/n-2)
#calculate absolute value of the Brown-Forsythe test statistic
t <- abs((D1-D2)/(s*sqrt(1/length(Group1)+1/length(Group2))))

#compare s to the critical value for any given alpha level to determine whether or not to conclude Ho: the error variance is constant
if (t>qt(0.975,118)) {sprintf("t larger than critical value - Reject Ho: error variance is not constant") } else {sprintf("t below critical value - Accept Ho: error variance is constant")}
#Or find p-value and if t is larger than this value, conclude that the error variance is not constant
if (pt(t,118,lower.tail=FALSE)<0.05) {sprintf("P-value is below a-value - Reject Ho: error variance is not constant") } else {sprintf("P-value is above a-value - Accept Ho: error variance is constant")}


###########################################################################
## Transformations in R

#The Box-Cox procedure chooses an optimal transformation to remediate deviations from the assumptions of the linear regression model.
library(MASS)
boxcox(College)
boxcox(College, lambda=seq(1,2,0.1))    #suggests that the best value of lambda is about 1.45

Data <- cbind(Data, Data$GPA^1.45)
names(Data)[dim(Data)[2]] <- "Yprime"
NewModel <- lm(Yprime ~ ACT, data=Data)

#still some departure from normality, it is much less pronounced
qqPlot(NewModel$residuals, dist="norm", col=palette()[1],
       ylab="Residual Quantiles", main="Normal Probability Plot", pch=19)
#plot residuals against the predictor ACT
plot(Data$ACT, NewModel$residuals, main="Residuals vs. Predictor", xlab="ACT Score", ylab="Residuals", pch=19)
abline(h=0)
#plot residuals against predicted values
plot(NewModel$fitted.values, NewModel$residuals, main="Fitted Values vs. Predictor", xlab="Fitted Values", ylab="Residuals", pch=19)
abline(h=0)

plot(Data$ACT, Data$Yprime, main = "Transformed Response", xlab="ACT Score",
     ylab="Transformed Freshman GPA", pch=20)
abline(NewModel$coefficients[1], NewModel$coefficients[2])


###########################################################################
## Conducting a Lack of Fit Test

Full <- lm(GPA ~ 0 + as.factor(ACT), data=Data)     #NOTE: Only works if there are values of the predictor which occur more than once

#compare two models (reduced and full)
anova(College, Full)
#Yields an F value of 0.8592 when sspe df = 99 and sslf df=19
#Use this information to test and see if model is a good fit to the data
if (0.8592>qf(0.95, 19, 99)) {sprintf("F*-value is above f-value(table) - Reject Ho: model is not a good fit to the data") } else {sprintf("F*-value is below f-value(table) - Accept Ho: model is a good fit to the data")}



###########################################################################
## Working with Matrices in R

#creating a matrix that is filled in column-by-column
Mcol <- matrix(1:6, ncol=3)
Mrow <- matrix(1:6, ncol=3, byrow=T)

#to multiply corresponding elements in matrices:
Mcol * Mcol
#to multiply matrices
Mcol %*% t(Mcol)

#to multiply vectors to get their scalar inner product
Data$residuals %*% Data$residuals

#matrix determinant
Msq <- matrix(1:9, ncol=3)
Msq[1,1] <- 254
det(Msq)
solve(Msq)

#Evaluated in linear regression to obtain the estimated regression parameters (compare to output from summary(College))
Y <- Data$GPA
X <- cbind(rep(1,length(Data$ACT)),Data$ACT)
b <- solve(t(X) %*% X) %*% t(X) %*% Y


###########################################################################
## Multiple Regression in R

Grocery <- read.table("C:/Users/davidryan/My Documents/DTR_Files/TrainingMaterials/R/LogisticRegressionCourse/Data/CH06PR09.txt",header=F)
names(Grocery) <- c("Hours", "Cases", "Costs", "Holiday")

#stem-and-leaf plots, boxplots, etc. to look for outlying observations, gaps in the data, and so on
#scatterplot matrix for all four variables
pairs(Grocery, pch=19)
#matrix enables you to tell whether the response variable appears to have any association with any 
#of the predictor variables, and if any two of the predictor variables appear to be correlated
#NOTE: Scatterplot matrix is not very helpful for the categorical variable "Holiday"

cor(Grocery)

#Fit a regression model for the three predictor variables Cases, Costs, Holiday and response variable Hours
Retailer <- lm(Hours ~ Cases+Costs+Holiday, data=Grocery)
summary(Retailer)

anova(Retailer)

SSTO <- sum(anova(Retailer)[,2])
MSE <- anova(Retailer)[4,3]

#To replicate the F* statistic found in summary
SSR <- sum(anova(Retailer)[1:3,2])
Fst <- (SSR/3) / MSE

#variance-covariance matrix for vector of parameter estimates
vcov(Retailer)

#To estimate mean response at a given vector:
X <- c(1, 24500, 7.40, 0)
Yhat <- t(X) %*% Retailer$coefficients

s <- sqrt(t(X) %*% vcov(Retailer) %*% X)

t <- qt(0.95, 48)

#90% confidence interval
c(Yhat - t*s, Yhat + t*s)

#find a prediction interval for a new observation, use:
spred <- sqrt(MSE + s^2)
c(Yhat - t*spred, Yhat + t*spred)

####### LACK OF FIT TEST #######################
Full2 <- lm(Hours ~ 0 + as.factor(Cases)+as.factor(Costs)+as.factor(Holiday), data=Grocery)     #NOTE: Only works if there are values of the predictor which occur more than once

#compare two models (reduced and full)
anova(Retailer, Full2)
#Yields an F value of 0.8592 when sspe df = 99 and sslf df=19
#Use this information to test and see if model is a good fit to the data
#if (0.8592>qf(0.95, 19, 99)) {sprintf("F*-value is above f-value(table) - Reject Ho: model is not a good fit to the data") } else {sprintf("F*-value is below f-value(table) - Accept Ho: model is a good fit to the data")}


Fullmin <- lm(Hours ~ Cases+Holiday, data=Grocery)
anova(Fullmin, Retailer)

summary(Fullmin)
Fullmin2 <- lm(Hours ~ Holiday, data=Grocery)
anova(Fullmin2, Retailer)

#still some departure from normality, it is much less pronounced
qqPlot(Retailer$residuals, dist="norm", col=palette()[1],
       ylab="Residual Quantiles", main="Retailer Normal Probability Plot", pch=19)

#still some departure from normality, it is much less pronounced
qqPlot(Fullmin2$residuals, dist="norm", col=palette()[1],
       ylab="Residual Quantiles", main="Finalized Retailer Normal Probability Plot", pch=19)




###########################################################################
## Extra Sum of Squares in R

SSR <- sum(anova(Retailer)[1:3,2])
MSR <- SSR/3
SSE <- anova(Retailer)[4,2]
MSE <- anova(Retailer)[4,3]

#obtain alternate decompositions of the regression sum of squares into extra 
#sum of squares by running new linear models with the predictors entered in a different order
Model2 <- lm(Hours ~ Holiday+Cases+Costs, data=Grocery)
anova(Model2)

#Suppose we want to test whether one or more of the variables can be dropped from the 
#original linear model (henceforth called the full model).  The easiest way to accomplish 
#this in R is to just run a new model that excludes the variables we are considering 
#dropping (henceforth called the reduced model), then perform a general linear test.  
Reduced <- lm(Hours ~ Cases+Holiday, data=Grocery)
anova(Reduced, Retailer)
if (0.3251>qf(0.95,1,48)) {sprintf("F*-value is above f-value(table) - Reject Ho: Reduced model is not acceptable; variables cannot be dropped") } else {sprintf("F*-value is below f-value(table) - Accept Ho: reduced model is acceptable; variables can be dropped from the model")}

Reduced2 <- lm (Hours - 600*Holiday ~ Cases, data=Grocery)
anova(Reduced2, Retailer)
#you will get an error message if you attempt to use the anova() function to compare this model with the full model, 
#because the two models do not have the same response variable.  Instead, you will need to obtain the SSE for this 
#reduced model, along with its degrees of freedom, from its ANOVA table, and the SSE from the full model, along with 
#its degrees of freedom, from the ANOVA table for the full model, then calculate F*

SSR_r <- sum(anova(Reduced2)[1,2])
MSR_r <- SSR/1
SSE_r <- anova(Reduced2)[2,2]
MSE_r <- anova(Reduced2)[2,3]

df <- anova(Retailer)[4,1]
df_r <- anova(Reduced2)[2,1]
Fstar <- ((SSE_r-SSE)/(df_r-df))/(SSE/df)
Ft <- qf(0.95,df_r-df,df)
if (Fstar>Ft) {sprintf("F*-value is above f-value(table) - Reject Ho: Reduced model is not acceptable; variables cannot be dropped") } else {sprintf("F*-value is below f-value(table) - Accept Ho: reduced model is acceptable; variables can be dropped from the model")}





###########################################################################
## Extra Sum of Squares in R

#To run a polynomial regression model on one or more predictor variables, it is advisable to first center the variables 
#by subtracting the corresponding mean of each, in order to reduce the intercorrelation among the variables
x1 <- Grocery$Cases - mean(Grocery$Cases)
x2 <- Grocery$Costs - mean(Grocery$Costs)

x1sq <- x1^2
x2sq <- x2^2
x1x2 <- x1*x2

Grocery <- cbind(Grocery, x1, x2, x1sq, x2sq, x1x2)

Poly <- lm(Hours ~ x1+x2+x1sq+x2sq+x1x2, data=Grocery)
summary(Poly)
#P-values for the t-tests for the parameter estimates corresponding to each predictor are each greater than 0.10, and so is the 
#P-value for the F-test, which tells us that all the predictors can be dropped from the model.  THIS MODEL IS NOT APPROPRIATE

Poly2 <- lm(Hours ~ x2+x2sq, data=Grocery)
summary(Poly2)
#THIS MODEL IS NOT APPROPRIATE EITHER
#plot the response variable against the centered predictor x2 in the usual manner
plot(Grocery$x2, Grocery$Hours, main="Polynomial Model", xlab="Costs(centered)", ylab="Hours", pch=19)
vec <- seq(-3,3,by=0.1)
lines(vec, Poly2$coefficients[1]+Poly2$coefficients[2]*vec+Poly2$coefficients[3]*vec^2)

#can also run an interaction model (8.29) involving all three of our predictor variables and the same response variable.  
#We will need to also center the third predictor, Holiday, and add that centered variable to the data table
x3 <- Grocery$Holiday - mean(Grocery$Holiday)
Grocery <- cbind(Grocery, x3)

Interact <- lm(Hours ~ x1+x2+x3+x1*x2+x1*x3+x2*x3, data=Grocery)
summary(Interact)
#appears that the only terms that should be kept in the model are x3, corresponding to Holiday, and maybe x1, depending 
#on the level of significance desired.  So an interaction model does not fit the data well at all.

#NOTE: to obtain a confidence interval for the mean response at some vector Xh, we have to make sure we first center Xh, since our model is based on centered predictors
#EX: For example, suppose we are using the second-order polynomial model above, with the single predictor Costs, and suppose we 
#are interested in a 95% confidence interval for the mean for the response Hours when Costs = 7.00
#Because our model consists of an intercept, a first-degree term, and a second-degree term, our vector Xh will consist of three elements: a one (for the intercept), 
#the centered version of Costs = 7.00, and the centered version of Costs = 7.00 squared.  That is,
X <- c(1, 7.00-mean(Grocery$Costs), (7.00-mean(Grocery$Costs))^2)





###########################################################################
## Model Selection in R

install.packages("leaps")
library(leaps)
#leaps() function will search for the best subsets of your predictors using whichever criterion you designate.   
#To use this function, we need to provide it with a matrix consisting of the predictor variables, a vector consisting 
#of the response variable, the names of the predictor variables, and the criterion to use

predVars <- Grocery[,2:4]
respVars <- Grocery[,1]
predVars_names <- names(predVars)

Lp <- leaps(x=predVars, y=respVars, names=predVars_names, method="Cp")    #Cp = Mallows' Cp criterion for model selection
#The best sub-model is that for which the Cp value is closest to p (the number of parameters in the model, including the intercept).  
#For the full model, we always have Cp = p.  The idea is to find a suitable reduced model, if possible.  Here the best reduced model 
#is the third one, consisting of Cases and Holiday, for which Cp = 2.325084 and p = 3.

Lp$which[which((Lp$Cp-Lp$size)==(min(Lp$Cp-Lp$size))),]

Lp_2 <- leaps(x=predVars, y=respVars, names=predVars_names, method="r2")    #r2 = r2 Criteria
Lp_3 <- leaps(x=predVars, y=respVars, names=predVars_names, method="adjr2")    #adjr2 = adjusted r2 Criteria

#highest value for either criteria indicates the best sub-model
Lp_2$which[which(head(Lp_2$r2,-1)==max(head(Lp_2$r2,-1))),]
Lp_3$which[which(head(Lp_3$adjr2,-1)==max(head(Lp_3$adjr2,-1))),]






#A less-attractive alternative to using the leaps() function would be to make a list of each sub-model you wish to consider, 
#then fit a linear model for each sub-model individually to obtain the selection criteria for that model
#we could start with our full model Retailer and delete just one variable, Costs.  Then we fit a new model named NewMod with only the remaining predictors
bestMod <- matrix(data=NA,nrow=3,ncol=4)
colnames(bestMod) <- c("Cp","AICp","SBCp","PRESSp")
NewMod <- update(Retailer, .~. -Costs)
summary(NewMod)   #MSR=0.6862; ARS=0.6734
n <- length(Grocery$Hours)
p <- dim(summary(NewMod)$coefficients)[1]
MSE <- anova(Retailer)[4,3]
SSE_nm <- anova(NewMod)[3,2]
Cp <- (SSE_nm/(MSE)) - (n-(2*p))
bestMod[1,1] <- Cp-p
bestMod[1,2] <- extractAIC(NewMod)[2]
bestMod[1,3] <- extractAIC(NewMod, k=log(n))[2]
bestMod[1,4] <- sum((NewMod$residuals/(1-hatvalues(NewMod)))^2)

NewMod <- update(NewMod, .~. -Cases)
summary(NewMod)   #MSR=0.657; ARS=0.6502
n <- length(Grocery$Hours)
p <- dim(summary(NewMod)$coefficients)[1]
MSE <- anova(Retailer)[4,3]
SSE_nm <- anova(NewMod)[2,2]
Cp <- (SSE_nm/(MSE)) - (n-(2*p))
bestMod[2,1] <- Cp-p
bestMod[2,2] <- extractAIC(NewMod)[2]
bestMod[2,3] <- extractAIC(NewMod, k=log(n))[2]
bestMod[2,4] <- sum((NewMod$residuals/(1-hatvalues(NewMod)))^2)

NewMod <- update(NewMod, .~. +Costs)
summary(NewMod)   #MSR=0.6581; ARS=0.6441
n <- length(Grocery$Hours)
p <- dim(summary(NewMod)$coefficients)[1]
MSE <- anova(Retailer)[4,3]
SSE_nm <- anova(NewMod)[3,2]
Cp <- (SSE_nm/(MSE)) - (n-(2*p))
bestMod[3,1] <- Cp-p
bestMod[3,2] <- extractAIC(NewMod)[2]
bestMod[3,3] <- extractAIC(NewMod, k=log(n))[2]
bestMod[3,4] <- sum((NewMod$residuals/(1-hatvalues(NewMod)))^2)

bestMod_abs <- abs(bestMod)
bestMod_log <- bestMod_abs[,1]==min(bestMod_abs[,1])
for (i in 2:dim(bestMod_abs)[2]) {
  temp <- bestMod_abs[,i]==min(bestMod_abs[,i])
  bestMod_log <- cbind(bestMod_log, temp)
}
colnames(bestMod_log) <- colnames(bestMod)



#STEPWISE REGRESSION
#Since there are 2p-1  subsets to consider among p-1 potential predictor variables, the above 
#process can become very tedious and time consuming when there are four or more predictors.  One way around this in R is to use stepwise regression. 
#R uses AICp criterion at each step
#FORWARD - fit a base model (one predictor) and a full model (all predictors to consider)
Base <- lm(Hours ~ Holiday, data=Grocery)

step(Base, scope=list(upper=Retailer, lower=~1), direction="forward", trace=FALSE)

step(Retailer, direction="backward", trace=FALSE)

step(Base, scope=list(upper=Retailer, lower=~1), direction="both", trace=FALSE)
step(Base, scope=list(upper=Retailer, lower=~1), direction="both", trace=TRUE)

#Backward stepwise regression using p-values to delete predictors one at a time
#use update functionality

#version of backward stepwise regression using R functions addterm() and dropterm() in MASS package
library(MASS)
dropterm(Retailer, test="F")
NewMod <- update(Retailer, .~. -Costs)
dropterm(NewMod, test="F")
#repeat the process until all F-values are larger than your F limit or all P-values are  below your alpha-level

#to use addterm() begin with a null model consisting of no predictors (just the intercept), use  ~1 in the model formula
Null <- lm(Hours ~ 1, data=Grocery)
addterm(Null, scope=Retailer, test="F")
NewMod <- update(Null, .~. +Holiday)
summary(NewMod)
addterm(NewMod, scope=Retailer, test="F")
NewMod <- update(NewMod, .~. +Cases)
summary(NewMod)
addterm(NewMod, scope=Retailer, test="F")
#repeat the process until all F-values are larger than your F limit or all P-values are  below your alpha-level
