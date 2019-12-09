# You only need to install packages once per machine
# (plus maybe after upgrading R), but otherwise they persist across R sessions.
install.packages('party')
install.packages('ROCR')

# Load the kyphosis data set.
require(rpart)

# Split randomly
x <- kyphosis[sample(1:nrow(kyphosis), nrow(kyphosis), replace = F),]
x.train <- kyphosis[1:floor(nrow(x)*.75), ]
x.evaluate <- kyphosis[(floor(nrow(x)*.75)+1):nrow(x), ]



# Create a model using "random forest and bagging ensemble algorithms
# utilizing conditional inference trees."
require(party)
x.model <- cforest(Kyphosis ~ Age + Number + Start, data=x.train,
                   control = cforest_unbiased(mtry = 3))

# Alternatively, use "recursive partitioning [...] in a conditional
# inference framework."
# x.model <- ctree(Kyphosis ~ Age + Number + Start, data=x.train)

# ctree plots nicely (but cforest doesn"t plot)
# plot (x.model)

# Use the model to predict the evaluation.
x.evaluate$prediction <- predict(x.model, newdata=x.evaluate)

rawData10_20_top9 <- rawData10_20[,colnames(rawData10_top9)[colnames(rawData10_top9)!="attackCatBinU2R"]]
rawData10_20_top9$prediction <- predict(mLogit, newdata=rawData10_20_top9)

#Refit model without NA variables
mLogit <- glm(attackCatBinU2R~dst_bytes+duration+src_bytes+root_shell+num_compromised+num_file_creations+hot, data=rawData10_top9, family="binomial")
rawData10_20_top9$prediction <- predict(mLogit, newdata=rawData10_20_top9, type="response")
rawData10_20_top9$attackCatBinU2R <- rawData10_20[,colnames(rawData10_top9)[colnames(rawData10_top9)=="attackCatBinU2R"]]

# Calculate the overall accuracy.
rawData10_20_top9$correct <- rawData10_20_top9$prediction == rawData10_20_top9$attackCatBinU2R
print(paste("% of predicted classifications correct", mean(rawData10_20_top9$correct)))

# Extract the class probabilities.
rawData10_20_top9$probabilities <- 1- unlist(treeresponse(mLogit,
                                                   newdata=rawData10_20_top9), use.names=F)[seq(1,nrow(rawData10_20_top9)*2,2)]

# Plot the performance of the model applied to the evaluation set as
# an ROC curve.
require(ROCR)
pred <- prediction(x.evaluate$probabilities, x.evaluate$Kyphosis)
perf <- performance(pred,"tpr","fpr")
plot(perf, main="ROC curve", colorize=T)

# And then a lift chart
perf <- performance(pred,"lift","rpp")
plot(perf, main="lift curve", colorize=T)
plot(perf, main="lift curve", colorize=F)


#Cumulative gains chart with baseline and ideal lines
plot(x=c(0.1, 1), y=c(1, 1), type="l", col="red", lwd=2,
     ylab="Lift", 
     xlab="% Customers Contacted",
     main = "Lift Chart",
     xlim=c(0.1,1), ylim=c(0,6))

#lines(x=c(1, 1, 1), y=c(1, 1, 1), col="darkgreen", lwd=2)
lines(x=unlist(perf@x.values), y=unlist(perf@y.values), col="orange", lwd=2)
legend("topright",c("Lift Curve","Baseline"),lty=c(1,1),lwd=c(2.5,2.5),col=c("orange",'red'))




library(rpart)
layout(matrix(c(1,2), 2, 1))
data(CCS)
CCS$Sample <- create.samples(CCS, est=0.4, val=0.4)
CCSEst <- CCS[CCS$Sample == "Estimation",]
CCS.glm <- glm(MonthGive ~ DonPerYear + LastDonAmt + Region + YearsGive,
               family=binomial(logit), data=CCSEst)
library(rpart)
CCS.rpart <- rpart(MonthGive ~ DonPerYear + LastDonAmt + Region + YearsGive,
                   data=CCSEst, cp=0.0074)
CCSVal <- CCS[CCS$Sample == "Validation",]
lift.chart(c("CCS.glm", "CCS.rpart"), data=CCSVal, targLevel="Yes",
           trueResp=0.01, type="cumulative", sub="Validation")
lift.chart(c("CCS.glm", "CCS.rpart"), data=CCSVal, targLevel="Yes",
           trueResp=0.01, type="incremental", sub="Validation")



























# Based on code from demo(ROCR)
library(ROCR)
mLogit <- glm(attackCatBinU2R~dst_bytes+duration+src_bytes+root_shell+num_compromised+num_file_creations+hot, data=rawData10_top9, family=binomial(logit))
summary(mLogit)
rawData10_20_top9 <- rawData10_20[,colnames(rawData10_top9)[colnames(rawData10_top9)!="attackCatBinU2R"]]
fitpreds <- predict(mLogit, newdata=rawData10_20_top9, types="response")
rawData10_20_top9$attackCatBinU2R <- rawData10_20[,colnames(rawData10_top9)[colnames(rawData10_top9)=="attackCatBinU2R"]]
rawData10_20_top9$prediction <- prediction(fitpreds, rawData10_20_top9$attackCatBinU2R)


rawData10_20_top9$performance <- performance(rawData10_20_top9$prediction, "tpr","fpr")
plot(rawData10_20_top9$performance, col='green', lwd=2, main="ROC Curve")
abline(a=0,b=1,lwd=2,lty=2,col='gray')



library(MASS)
basemodel <- lm(attackCatBinU2R~dst_bytes+duration+src_bytes+root_shell+num_compromised+num_file_creations+hot, data=rawData10_top9)
norm1 <- eval(stepAIC(basemodel,
                      scope=list(lower=.~ 1,upper=.~(dst_bytes+duration+src_bytes+root_shell+num_compromised+num_file_creations+hot)^5),
                      k=2)$call) # k=2 for AIC
norm1$call
# lm(formula = z ~ x1 + x3 + x4, data = schools)
norm2 <- eval(stepAIC(basemodel,
                      scope=list(lower=.~ 1,upper=.~(dst_bytes+duration+src_bytes+root_shell+num_compromised+num_file_creations+hot)^5),
                      k=log(20))$call) # k=log(sample size) for BIC
norm2$call
# lm(formula = z ~ x3 + x4, data = schools)







# Calculate the overall accuracy.
rawData10_20_top9$correct <- rawData10_20_top9$prediction == rawData10_20_top9$attackCatBinU2R
print(paste("% of predicted classifications correct", mean(rawData10_20_top9$correct)))

# Extract the class probabilities.
rawData10_20_top9$probabilities <- 1- unlist(treeresponse(mLogit,
                                                          newdata=rawData10_20_top9), use.names=F)[seq(1,nrow(rawData10_20_top9)*2,2)]






data(ROCR.hiv)
pp <- ROCR.hiv$hiv.svm$predictions
ll <- ROCR.hiv$hiv.svm$labels
pred <- prediction(pp, ll)
perf <- performance(pred, "tpr", "fpr")
pdf("graphics/rplot-rocr-4plots.pdf")
par(mar = c(5,4,4,2)+0.1)
par(mfrow = c(2, 2))
plot(perf, avg = "threshold", colorize = T, lwd = 3,
     main = "Standard ROC curve.")

plot(perf, lty = 3, col = "grey78", add = T)
perf <- performance(pred, "prec", "rec")
plot(perf, avg = "threshold", colorize = T, lwd = 3,
     main = "Precision/Recall graph.")

plot(perf, lty = 3, col = "grey78", add = T)
perf <- performance(pred, "sens", "spec")
plot(perf, avg = "threshold", colorize = T, lwd = 3,
     main = "Sensitivity/Specificity plot.")

plot(perf, lty = 3, col = "grey78", add = T)
perf <- performance(pred, "lift", "rpp")
plot(perf, avg = "threshold", colorize = T, lwd = 3,
     main = "Lift chart.")

plot(perf, lty = 3, col = "grey78", add = T)
dev.off()




















# Based on code from demo(ROCR)
library(ROCR)
data(ROCR.hiv)
pp <- ROCR.hiv$hiv.svm$predictions
ll <- ROCR.hiv$hiv.svm$labels
pred <- prediction(pp, ll)
perf <- performance(pred, "tpr", "fpr")
pdf("graphics/rplot-rocr-4plots.pdf")
par(mar = c(5,4,4,2)+0.1)
par(mfrow = c(2, 2))
plot(perf, avg = "threshold", colorize = T, lwd = 3,
     main = "Standard ROC curve.")

plot(perf, lty = 3, col = "grey78", add = T)
perf <- performance(pred, "prec", "rec")
plot(perf, avg = "threshold", colorize = T, lwd = 3,
     main = "Precision/Recall graph.")

plot(perf, lty = 3, col = "grey78", add = T)
perf <- performance(pred, "sens", "spec")
plot(perf, avg = "threshold", colorize = T, lwd = 3,
     main = "Sensitivity/Specificity plot.")

plot(perf, lty = 3, col = "grey78", add = T)
perf <- performance(pred, "lift", "rpp")
plot(perf, avg = "threshold", colorize = T, lwd = 3,
     main = "Lift chart.")

plot(perf, lty = 3, col = "grey78", add = T)
dev.off()