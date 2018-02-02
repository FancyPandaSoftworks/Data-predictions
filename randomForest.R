#reset
rm(list =ls())
cat('\014')


library(randomForest)
library("party")
library(tibble)
library(ggplot2)
library(caret)
library(caTools)


myData <- read.csv("students.data.csv")
predictData <- read.csv("predict.students.data.csv")
####################################################

myData$Address<-NULL
myData$Guardian<-NULL
myData$Pass<- as.factor(myData$Pass)
#predictData cleansing
predictData$Identifier<-NULL
predictData$Address<-NULL
predictData$Guardian<-NULL
myData$Family_size<- NULL
myData$Employment_mother<-NULL
myData$Age<-NULL
myData$Cohab_status<- NULL
myData$Guardian <- NULL
myData$Paid<- NULL
myData$Higher<- NULL
#--------------------
predictData$Family_size<- NULL
predictData$Employment_mother<-NULL
predictData$Age<-NULL
predictData$Cohab_status<- NULL
predictData$Guardian <- NULL
predictData$Paid<- NULL
predictData$Higher<- NULL
################################################


#training and validation
label <- as.numeric(myData[,23])
sample <- sample.split(label, SplitRatio = 0.7)

y <- 296
for (x in 1:nrow(myData)) {
  if(myData$Pass[x]==FALSE){
    myData[y,]<- myData[x,]
    y<- y+1
  }
  
}
train <- subset(myData, sample==TRUE)

#Create testing data
test <- subset(myData, sample==FALSE)

################################################


#Making random forest with the variables
output.forest<- randomForest(Pass ~ Primary_school + Gender + Education_mother+ Education_father + Employment_father + 
                             Choice_of_school + Traveltime + Studytime + Failures + Educ_support
                             + Family_educ_support + Activities + Nursery + Internet + Romantic + Family_relationships
                             + Freetime + Go_out + Day_alcohol + Weekend_alcohol + Health + Absences,
                             #+ myData[,29]+ myData[,30], 
                             data = train, importance = TRUE, ntree = 2000)


print(output.forest)
importance(output.forest)
varImpPlot(output.forest)

################################################
#Testing
prediction <- predict(output.forest, newdata = test, type = "class")

#prediction data 
prediction2 <- predict(output.forest, newdata = predictData, type = "class")
predictData <- read.csv("predict.students.data.csv")
prediction2 <- as.data.frame(prediction2)
answer <- cbind(predictData$Identifier, prediction2)

#Show F1 score
result <-confusionMatrix(prediction, test$Pass)
result$byClass[7]
#F1 between 0.79 and 0.87
write.table(answer, "answer.txt")
