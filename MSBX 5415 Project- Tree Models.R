rm(list = ls())
setwd("C:/Users/danie/downloads")
load("bar_clean_final.rdata")

bardata = bar

#pred function for testing models

prf <- function(predAct){
  ## predAct is two col dataframe of pred,act
  preds = predAct[,1]
  trues = predAct[,2]
  xTab <- table(preds, trues)
  clss <- as.character(sort(unique(preds)))
  r <- matrix(NA, ncol = 7, nrow = 1, 
              dimnames = list(c(),c('Acc',
                                    paste("P",clss[1],sep='_'), 
                                    paste("R",clss[1],sep='_'), 
                                    paste("F",clss[1],sep='_'), 
                                    paste("P",clss[2],sep='_'), 
                                    paste("R",clss[2],sep='_'), 
                                    paste("F",clss[2],sep='_'))))
  r[1,1] <- sum(xTab[1,1],xTab[2,2])/sum(xTab) # Accuracy
  r[1,2] <- xTab[1,1]/sum(xTab[,1]) # Miss Precision
  r[1,3] <- xTab[1,1]/sum(xTab[1,]) # Miss Recall
  r[1,4] <- (2*r[1,2]*r[1,3])/sum(r[1,2],r[1,3]) # Miss F
  r[1,5] <- xTab[2,2]/sum(xTab[,2]) # Hit Precision
  r[1,6] <- xTab[2,2]/sum(xTab[2,]) # Hit Recall
  r[1,7] <- (2*r[1,5]*r[1,6])/sum(r[1,5],r[1,6]) # Hit F
  r}


# building basic tree

library(tree)
tree.bar = tree(as.factor(pass_bar) ~. -p.bar1 - p.bar2 -ID, data = bardata, 
                control = tree.control(nobs = nrow(bardata), mindev = 0.001))
summary(tree.bar)
plot(tree.bar)
text(tree.bar, pretty = 0)
bar.pred = predict(tree.bar, type = "class")
bar.pred <- predict(tree.bar)

#evaluate performance
set.seed(200)
train = sample(nrow(bardata), 0.6 * nrow(bardata))
bar.test = bardata[-train, ]
tree.bar.train = tree(as.factor(pass_bar)~. -p.bar1 - p.bar2 -ID, data = bardata, subset = train,
                      control = tree.control(nobs = nrow(bardata), mindev = 0.001))

bar.tree.pred <- predict(tree.bar.train, newdata = bar.test, type = "class")
table (bar.test$pass_bar, bar.tree.pred)
#run evaluation function 
predAct_base <- data.frame(bar.tree.pred, bar.test$pass_bar)
prf(predAct_base)

# cross validation
set.seed(200)
tree.train.bar <- tree(as.factor(pass_bar) ~. -p.bar1 - p.bar2 -ID, data = bardata, subset = train,
                       control = tree.control(nobs = nrow(bardata), mindev = 0.001))
cv.bar <- cv.tree(tree.bar.train, FUN = prune.tree)
cv.bar
plot(cv.bar$size, cv.bar$dev)
plot(cv.bar$k, cv.bar$dev)

#prune from cross validation results 
prune.bar <- prune.tree(tree.train.bar, best = 6)
prune.bar
plot(prune.bar)
text(prune.bar, pretty = 0) 

#evaluate pruned tree
prune.pred <- predict(prune.bar, newdata = bar.test, type = "class")
table (bar.test$pass_bar,prune.pred)
#run evaluation function 
predAct_base <- data.frame(prune.pred, bar.test$pass_bar)
prf(predAct_base)



# gbm Model

#install.packages('gbm') 
library(gbm)
set.seed(300)

#All are very close but 2000 trees with threshold of .2 for prediction had highest accuracy 

gbm.boost.bar <- gbm(pass_bar ~. -p.bar1 - p.bar2 -ID, data = bardata[train, ],
                     distribution = "bernoulli", n.trees = 2000)

summary(gbm.boost.bar)

pred.gbm.boost <- predict(gbm.boost.bar, newdata = bar.test, n.trees = 2000, type = "response")

table(bar.test$pass_bar, pred.gbm.boost> 0.2)
#run evaluation function
predAct_base <- data.frame(pred.gbm.boost > 0.2, bar.test$pass_bar)
prf(predAct_base)


# others I tried:

#1000 trees:
gbm.boost.bar <- gbm(pass_bar ~. -p.bar1 - p.bar2 -ID, data = bardata[train,],
                     distribution = "bernoulli", n.trees = 1000)
pred.gbm.boost <- predict(gbm.boost.bar, newdata = bar.test, n.trees = 1000, type = "response")

table(bar.test$pass_bar, pred.gbm.boost> 0.2)
predAct_base <- data.frame(pred.gbm.boost > 0.2, bar.test$pass_bar)
prf(predAct_base)


# 5000 trees:
gbm.boost.bar <- gbm(pass_bar ~. -p.bar1 - p.bar2 -ID, data = bardata[train,],
                     distribution = "bernoulli", n.trees = 5000)
pred.gbm.boost <- predict(gbm.boost.bar, newdata = bar.test, n.trees = 5000, type = "response")
table(bar.test$pass_bar, pred.gbm.boost> 0.2)

predAct_base <- data.frame(pred.gbm.boost > 0.2, bar.test$pass_bar)
prf(predAct_base)




#Adaboost model 
#install.packages('adabag') 
#install.packages('caret')
library(adabag)
library(caret)

set.seed(200)
#re did split because requires dataframe for training and test sets 
parts = createDataPartition(bardata$pass_bar, p = 0.6, list = F)
ada.train = bardata[parts, ]
ada.test = bardata[-parts, ]
#converted to factor outside of ada model because model can not handle it inside
ada.train$pass_bar = factor(ada.train$pass_bar)


bar.adaboost= boosting(pass_bar~. -p.bar1 - p.bar2 -ID, 
                       data = ada.train, boos = T, mfinal = 100)
summary(bar.adaboost)
importanceplot(bar.adaboost)

ada.pred <- predict(bar.adaboost, ada.test)
ada.pred
predAct_base <- data.frame(ada.pred, bar.test$pass_bar)
prf(predAct_base)

# evaluation function could not accept ada.pred so calculated accuracy and F1_0 below
Accuracy = (76 + 7697)/(76+61+348+7697)
precision_0 = 76 / (76 + 61)
recall_0 = 76 / (76 + 348)
F1_0 = (2 * precision_0 * recall_0)/(precision_0 +recall_0)

Accuracy
F1_0

