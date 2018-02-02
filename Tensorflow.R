#reset
rm(list =ls())
cat('\014')

#libraries
library(caret)
library(nnet)
library(caTools)
library(tensorflow)
answer<- NULL
  
#import data
myData <- read.csv("students.data.csv")
testData<- read.csv("predict.students.data.csv")
#myData cleansing
myData$Identifier<-NULL
myData$Address<-NULL
myData$Guardian<-NULL
myData$Pass<- as.factor(myData$Pass)
myData$Pass <- as.numeric(myData$Pass)
#testdata cleansing
testData$Identifier<-NULL
testData$Address<-NULL
testData$Guardian<-NULL
###############################################
#Testing what happens when removing more variables
myData$Family_size<- NULL
myData$Employment_mother<-NULL
myData$Age<-NULL
myData$Cohab_status<- NULL
myData$Guardian <- NULL
myData$Paid<- NULL
myData$Higher<- NULL
#--------------------
testData$Family_size<- NULL
testData$Employment_mother<-NULL
testData$Age<-NULL
testData$Cohab_status<- NULL
testData$Guardian <- NULL
testData$Paid<- NULL
testData$Higher<- NULL


################################################

#Convert all non numeric data
for (x in 1:length(myData)) {
  if(class(myData[1,x])=="factor"){
    myData[,x]<- as.numeric(myData[,x])
  }
  
}
for (x in 1:length(testData)) {
  if(class(testData[1,x])=="factor"){
    testData[,x]<- as.numeric(testData[,x])
  }
  
}
#y <- 296
#for(i in 1:length(myData)){
#  if(myData$Pass[i]==FALSE){
#    myData[y,]<- myData[i,]
#    y<-y+1
#  }
#  
#}


#Label(Passing or failing)
label <- as.numeric(myData[,23])
#Make it a matrix for TF
myData = as.matrix(myData[,1:22])
testData = as.matrix(testData[,1:22])
#Create training data
#NOTE! this is just in 1 sheets, there is an external testing data for later
############################################################################
sample <- sample.split(label, SplitRatio = 0.7)
train <- subset(myData, sample==TRUE)
trainLabel <- subset(label, sample==TRUE)
#Create testing data
test <- subset(myData, sample==FALSE)
testLabel <- subset(label, sample == FALSE)
############################################################################
train <- myData
trainLabel <- label
#Define layer
add_layer <- function(x, in_size, out_size, act){
  w = tf$Variable(tf$random_normal(shape(in_size, out_size)))
  b = tf$Variable(tf$random_normal(shape(1, out_size)))
  wxb = tf$matmul(x,w)+ b
  y = act(wxb)
  return(y)
  
}
#Create layer
x = tf$placeholder(tf$float32, shape(NULL,22))#22 is the total parameters
ty = tf$placeholder(tf$float32, shape(NULL, 2))# 2 is the output
l1 = add_layer(x, 22, 45, tf$nn$relu)
l = add_layer(l1, 45,2, tf$nn$softmax)
#cost function for cross valuidation
cross_entropy = tf$reduce_mean(-tf$reduce_sum(ty*tf$log(l+1e-9), reduction_indices = 1L))
#cross_entropy = tf$reduce_mean(-tf$reduce_sum(ty*tf$log(softmax), reduction_indices = 1L))

#Optimization
optimizer <- tf$train$MomentumOptimizer(0.01, 0.01)
train_step <- optimizer$minimize(cross_entropy)
####################################################################################
init=tf$global_variables_initializer()
sess = tf$Session()
sess$run(init)
####################################################################################
#training

for(i in 1:10000){
  sess$run(train_step, feed_dict = dict(x =train, ty=class.ind(trainLabel)))
}

round(sess$run(l, feed_dict = dict(x = myData)), 2)

#testing
predictLabel = NULL
for(i in 1:nrow(test)){
  predictLabel[i]=which.max(sess$run(l, feed_dict = dict(x= test))[i,])
}
acc = mean(predictLabel==testLabel)
acc

conTable = table(testLabel, predictLabel)
conTable
##################################################################################
predictLabel2 = NULL
for(i in 1:nrow(testData)){
  predictLabel2[i]=which.max(sess$run(l, feed_dict = dict(x= testData))[i,])
}

predictLabel3 = NULL
for(i in 1:nrow(myData)){
  predictLabel3[i]=which.max(sess$run(l, feed_dict = dict(x= myData))[i,])
}
##################################################################################
#getting the answer
predictLabel2<- as.data.frame(predictLabel2)
for (x in 1:nrow(predictLabel2)) {
  if(predictLabel2[x,1]==1)
  {
    predictLabel2[x,1]<- "FALSE"
  }
  else{
    predictLabel2[x,1]<-"TRUE"
  }
  
  
}
testData<- read.csv("predict.students.data.csv")
answer<- cbind(testData, predictLabel2)
answer<- answer[-c(1:30)]
write.table(answer, "answer.txt")
#################################################################################
#F1_score
result <- confusionMatrix(testLabel, predictLabel)
result$byClass[7]

