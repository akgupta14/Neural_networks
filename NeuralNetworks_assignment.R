
startup_data <- read.csv(file.choose())

head(startup_data)
dim(startup_data)
library(Hmisc)
describe(startup_data)

temp <- startup_data[,-4]
head(temp)

pairs(temp)

#R&D Spend and Profit shows perfect positive correlation. Marketing Spend also shows a positive correction with Profit.
#R&D spend and Marketing spend also shows a positive correlation

nrow(startup_data)- sum(complete.cases(startup_data))


library(ggcorrplot)

ggcorrplot(temp,type="lower",outline.color = "black",lab=T,
           ggtheme = ggplot2::theme_gray,method = "circle")

library(DataExplorer)

plot_histogram(temp)

plot_boxplot(temp,by="Profit")

#For low values of Marketing spend ,profit is less. Similarly as the spend on R&D increases , the profit also increases.

summary(temp)

model <- lm(Profit~.,temp)

summary(model)

#R&D spend is the most significant variable

# custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}



startup_norm <- as.data.frame(lapply(temp, normalize))

head(startup_norm)


#Create Train and Test data

ran <- sample(1:nrow(startup_norm),0.7*nrow(startup_norm))

startup_train <- startup_norm[ran,]
startup_test <- startup_norm[-ran,]

head(startup_train)
dim(startup_train)

head(startup_test)
dim(startup_test)

library(neuralnet)

#Simple ANN with single Neuron

startup_model <- neuralnet(formula=Profit~.,startup_train,linear.output = F)
startup_model$net.result
startup_model$covariate
startup_model$result.matrix


model_results <- compute(startup_model, startup_test[1:3])
predicted_strength <- model_results$net.result
cor(predicted_strength, startup_test$Profit)

#########Accuracy with simple neuron is 96%###################

##ANN with hidden 
startup_model <- neuralnet(formula=Profit~.,data=startup_train,linear.output = F,hidden = c(5,2))


startup_model$net.result
startup_model$covariate
startup_model$result.matrix

windows()
plot(startup_model)

model_results <- compute(startup_model, startup_test[1:3])

predicted_strength <- model_results$net.result
cor(predicted_strength, startup_test$Profit)

#####Accuracy is 96%#####################

out <- cbind(startup_model$covariate,startup_model$net.result[[1]])
dimnames(out)<-list(NULL,c("R.D.Spend","Administration","Marketing spend","model results"))

View(out)

############################################################################################################

forestfires_data <- read.csv(file.choose())

head(forestfires_data)
dim(forestfires_data)

table(forestfires_data$month)

table(forestfires_data$month,forestfires_data$size_category)
table(forestfires_data$day,forestfires_data$size_category)

table(forestfires_data$month,forestfires_data$day,forestfires_data$size_category)

table(forestfires_data$FFMC,forestfires_data$size_category)
barplot(table(forestfires_data$FFMC))

table(forestfires_data$DMC,forestfires_data$size_category)
barplot(table(forestfires_data$DMC))

table(forestfires_data$DC,forestfires_data$size_category)
barplot(table(forestfires_data$DC))

table(forestfires_data$ISI,forestfires_data$size_category)
barplot(table(forestfires_data$ISI))

table(forestfires_data$temp,forestfires_data$size_category)
barplot(table(forestfires_data$temp))

table(forestfires_data$RH,forestfires_data$size_category)
barplot(table(forestfires_data$RH))

table(forestfires_data$wind,forestfires_data$size_category)

table(forestfires_data$rain,forestfires_data$size_category)

table(forestfires_data$area,forestfires_data$size_category)

prop.table(table(forestfires_data$size_category))
#EDA shows month,rain,wind, area,temp have a strong correlation with Size category-burnt area of the forest

#Removing columns month and day since the dummy variables are created for the same

forest_newdata <- forestfires_data[,-c(1,2)]
dim(forest_newdata)

library(Hmisc)
describe(forest_newdata)

head(forest_newdata)


##########PCA ############################

colnames(forest_newdata)
head(forest_newdata)

normalize<-function(x){return((x-min(x))/(max(x)-min(x)))}
#create normalize function
forest_scaled<-as.data.frame(lapply(forest_newdata[1:28],normalize ))
head(forest_scaled)

pca <- princomp(forest_scaled,cor=T,scores = T,covmat = NULL)

summary(pca)

####Attach 1st 4 PC scores to the data
pca_data3 <- cbind(forest_scaled,pca$scores[,1:4])

pca_data3 <- cbind(pca_data3,forest_newdata$size_category)

library(reshape)

#Change column name to Type
pca_data3 <- rename(pca_data3,c('forest_newdata$size_category' = "size_category"))

head(pca_data3)
colnames(pca_data3)
str(pca_data3)

library(ggplot2)
ggplot(pca_data3,aes(x=Comp.1,y=Comp.2,color=size_category,shape=size_category))+
  geom_hline(yintercept = 0,lty=2)+
  geom_vline(xintercept = 0,lty=2)+
  geom_point(alpha=0.8)

###Using PCA we can see that data seems noisy

#####Feature selection using Random Forest#####################

library(randomForest)

model <- randomForest(forest_newdata$size_category~.,data = forest_newdata,ntree=1000)

importance(model)

###As per variable importance we can remove variables like area,monthapr,monthdec,
###monthjun,monthmay,monthoct
colnames(forest_newdata)


#Simple ANN with single Neuron

######To build ANN algorithm prepare data excl PCA scores######

colnames(pca_data3)
nnet_data <- pca_data3[,-c(29:32)]
head(nnet_data)


library(caret)
intraininglocal <- createDataPartition(nnet_data$size_category,p=0.7,list = F)

forest_training <- nnet_data[intraininglocal,]
forest_Testing <- nnet_data[-intraininglocal,]

head(forest_training)
str(forest_training)
dim(forest_training)
dim(forest_Testing)

#####Build neuralnet model########################

####Since we have to predict class which is a factor variable we will use nnet()
library(nnet)

str(forest_Testing)

forest_model <- nnet(size_category~.,forest_training,size=1)

install.packages("NeuralNetTools")
library(NeuralNetTools)
plotnet(forest_model,alpha=0.6)

colnames(forest_Testing)

###The "compute" function then creates the prediction variable
model_results <- compute(forest_model, forest_Testing[1:28])

#A "results" variable compares the predicted data with the actual data

results <- data.frame(actual=forest_Testing$size_category,prediction=model_results$net.result)

results

#######To check the accuracy on test data################

colnames(forest_Testing)
newprediction1 <- predict(forest_model,type="class",newdata=forest_Testing[,-29])
newprediction1

str(newprediction1)

newprediction1 <- as.factor(newprediction1)

head(forest_Testing)
confusionMatrix(newprediction1,forest_Testing$size_category)

#####96% accuracy ########################

#####Model with 2 hidden layers########################
forest_model2 <- nnet(size_category~.,forest_training,size=2)

newprediction2 <- predict(forest_model2,type="class",newdata=forest_Testing[,-29])
newprediction2

newprediction2 <- as.factor(newprediction2)

confusionMatrix(newprediction2,forest_Testing$size_category)

#####Model with 1 hidden layer gives the best accuracy

install.packages("Metrics")
library(Metrics)

totalError <- c()

cv<-10

###Divide train data in 10 equal portions
nrow(forest_training)
cvDivider <- floor(nrow(forest_training)/(cv+1))
cvDivider

datasetIndex <- c((cv*cvDivider):(cv*cvDivider+cvDivider))

dataTest <- forest_training[datasetIndex,]

##Everything else to train
dataTrain <- forest_training[-datasetIndex,]

dim(dataTest)
dim(dataTrain)

###Using bootstrap method to test the accuracy

acc <- c()
for (cv in seq(1:cv))
{
  
  datasetIndex <- c((cv*cvDivider):(cv*cvDivider+cvDivider))
  
  dataTest <- forest_training[datasetIndex,]
  
  ##Everything else to train
  dataTrain <- forest_training[-datasetIndex,]
 
  forest_model <- nnet(size_category~.,dataTrain,size=1,maxit=500,trace=T)
  pred <- predict(forest_model,type="class",newdata=dataTest[,-29])
  

  a<- table(dataTest$size_category,pred)
  acc <- c(acc,sum(diag(a))/sum(a))
  
}

summary(acc)

####Mean accuracy is 62% and median accuracy is 92%


head(forest_training)
#######Using neuralnet, we will categorise size_category in binary format

nnet_foresttrain <- forest_training
nnet_foresttrain <- cbind(nnet_foresttrain,forest_training$size_category == 'large')
nnet_foresttrain <- cbind(nnet_foresttrain,forest_training$size_category == 'small')

View(nnet_foresttrain)
colnames(nnet_foresttrain)

names(nnet_foresttrain)[30] <- 'large'
names(nnet_foresttrain)[31] <- 'small'

colnames(nnet_foresttrain)

nnet_foresttrain[29] <- NULL

colnames(nnet_foresttrain)

library(neuralnet)
nn <- neuralnet(large+small~.,data=nnet_foresttrain,hidden = 1) 

plot(nn)
colnames(forest_Testing)
mypredict <- compute(nn,forest_Testing[,-29])$net.result

# Put multiple binary output to categorical output
maxidx <- function(arr) {
  return(which(arr == max(arr)))
}
idx <- apply(mypredict, c(1), maxidx)
prediction <- c('large', 'small')[idx]
table(prediction, forest_Testing$size_category)

(38+110)/155

######Accuracy is 95% using neuralnet() for 1 hidden layer

######Let's try for 2 hidden layers

nn <- neuralnet(large+small~.,data=nnet_foresttrain,hidden = c(8,2)) 

plot(nn)
colnames(forest_Testing)
mypredict <- compute(nn,forest_Testing[,-29])$net.result

# Put multiple binary output to categorical output
maxidx <- function(arr) {
  return(which(arr == max(arr)))
}
idx <- apply(mypredict, c(1), maxidx)
prediction <- c('large', 'small')[idx]
table(prediction, forest_Testing$size_category)

37+117

(35+111)/154

######Accuracy is 94% with 2 hidden layers

###############################################################################################

# Load the Concrete data

concrete_data <- read.csv("D:/R Excel Sessions/Assignments/Neural Networks/concrete.csv")

head(concrete_data)
library(caret)


# custom normalization function
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}
##### Neural Networks -------------------# Load the Concrete data# custom normalization functionnormalize <- function(x) {   return((x - min(x)) / (max(x) - min(x)))}

concrete_norm <- as.data.frame(lapply(concrete_data, normalize))

head(concrete_norm)

# create training and test data
concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]
## Training a model on the data ----
# train the neuralnet model

install.packages("neuralnet")
library(neuralnet)


# simple ANN with only a single hidden neuron

concrete_model <- neuralnet(formula = strength ~ cement + slag +
                              ash + water + superplastic + 
                              coarseagg + fineagg + age,
                            data = concrete_train)
# visualize the network topology
windows()
plot(concrete_model)

## Evaluating model performance ----
# obtain model results
colnames(concrete_test)
model_results <- compute(concrete_model, concrete_test[1:8])
# obtain predicted strength values
predicted_strength <- model_results$net.result
# examine the correlation between predicted and actual values
cor(predicted_strength, concrete_test$strength)

## Improving model performance ----
# a more complex neural network topology with 5 hidden neurons
concrete_model2 <- neuralnet(strength ~ cement + slag +
                               ash + water + superplastic + 
                               coarseagg + fineagg + age,
                             data = concrete_train, hidden =c(5,2))
# plot the network
windows()
plot(concrete_model2)
# evaluate the results as we did before
head(concrete_test)
model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result

print(model_results2)
head(concrete_test)

cor(predicted_strength2, concrete_test$strength)

########Using PCA#############################################

pca <- princomp(concrete_norm,cor=T,scores = T,covmat = NULL)

summary(pca)

####Attach 1st 4 PC scores to the data
pca_data <- cbind(concrete_norm,pca$scores[,1:4])

head(pca_data)


library(ggplot2)
ggplot(pca_data,aes(x=Comp.1,y=Comp.2,color=strength))+
  geom_hline(yintercept = 0,lty=2)+
  geom_vline(xintercept = 0,lty=2)+
  geom_point(alpha=0.8)

head(pca_data)

ran <- sample(1:nrow(pca_data),0.7*nrow(pca_data))

pca_train <- pca_data[ran,]
pca_test <- pca_data[-ran,]

head(pca_train)
dim(pca_train)

head(pca_test)
dim(pca_test)

###########Single ANN using PCA scores############################
concrete_pcamodel <- neuralnet(formula = strength ~ Comp.1+Comp.2+Comp.3+Comp.4,
                            data = pca_train)
# visualize the network topology
windows()
plot(concrete_pcamodel)

## Evaluating model performance ----
# obtain model results
colnames(pca_test)
model_pcaresults <- compute(concrete_pcamodel, pca_test[10:13])
# obtain predicted strength values
pcapredicted_strength <- model_pcaresults$net.result
# examine the correlation between predicted and actual values
cor(pcapredicted_strength, pca_test$strength)

######With 5 hidden neurons

concrete_pcamodel2 <- neuralnet(formula = strength ~ Comp.1+Comp.2+Comp.3+Comp.4,
                               data = pca_train, hidden=c(5,2))

plot(concrete_pcamodel2)

## Evaluating model performance ----
# obtain model results
colnames(pca_test)
model_pcaresults2 <- compute(concrete_pcamodel2, pca_test[10:13])
# obtain predicted strength values
pcapredicted_strength2 <- model_pcaresults2$net.result
# examine the correlation between predicted and actual values
cor(pcapredicted_strength2, pca_test$strength)

########Model performance has improved with PCA##############################

