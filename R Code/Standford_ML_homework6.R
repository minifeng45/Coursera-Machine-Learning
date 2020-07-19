#### Support Vector Machines ####
### 1-1 Example Dataset 1 
#library to read matlab data formats into R
library(R.matlab)
library(e1071)
library(ggplot2)
library(dplyr)
dat = readMat("Desktop/programing/R/Coursera machine learning/Homework/machine-learning-ex6/ex6/ex6data1.mat")
dat.for.plot = data.frame(x1 = dat$X[,1], 
                             x2 = dat$X[,2],
                             y = dat$y)
fig1 = ggplot(dat.for.plot,aes(x = x1, y = x2, color = as.factor(y)))+
  geom_point()

fig1
# SVM
SVM.plot = function(cost =1){
  
m = svm(y ~x1+x2,cost=cost,
        type = "C-classification", 
        kernel = "linear", 
        data = dat.for.plot,
        scale = F)


scatter.plot = ggplot(dat.for.plot,aes(x = x1, y = x2, color = as.factor(y)))+
  geom_point()
layout.plot =  scatter.plot + 
  geom_point(data = dat.for.plot[m$index, ], aes(x = x1, y = x2), color = "purple", size = 4, alpha = 0.5)

#calculate slope and intercept of decision boundary from weight vector and svm model
w = t(m$coefs) %*% m$SV
slope_1 = -w[1]/w[2]
intercept_1 = m$rho/w[2]

#build scatter plot of training dataset
scatter_plot <- ggplot(data = dat.for.plot, aes(x = x1, y = x2, color = as.factor(y))) + 
  geom_point() + scale_color_manual(values = c("red", "blue"))
#add decision boundary
plot_decision <- scatter_plot + geom_abline(slope = slope_1, intercept = intercept_1) 
#add margin boundaries
fig <- plot_decision + 
  geom_abline(slope = slope_1 , intercept = intercept_1- 1/w[2], linetype = "dashed")+
  geom_abline(slope = slope_1 , intercept = intercept_1+ 1/w[2], linetype = "dashed")
#display plot
return(fig)
}

fig2 =  SVM.plot(cost =1)
fig2 
fig3 = SVM.plot(cost =100)
fig3 
####1.2 SVM with Gaussian Kernels####
## 1.2.1 Gaussian Kernel##
x1 = c(1,2,1)
x2 = c(0,4,-1)
sigma = 2 
gus.numerator = -sum((x1-x2)^2)
gus.denominator = 2*sigma^2
ans1 = exp(gus.numerator/gus.denominator)
ans1

## Example Dataset 2##
dat = readMat("Desktop/programing/R/Coursera machine learning/Homework/machine-learning-ex6/ex6/ex6data2.mat")
dat.for.plot = data.frame(x1 = dat$X[,1], 
                          x2 = dat$X[,2],
                          y = dat$y)
fig4 = ggplot(dat.for.plot,aes(x = x1, y = x2, color = as.factor(y)))+
  geom_point()

fig4
## 1.2.2 Example Dataset 2 ##
# installing library ElemStatLearn 
SVM.gus = function(cost, sigma){
m = svm(y ~x1+x2,cost=cost,
        type = "C-classification", 
        kernel = "radial", 
        sigma = sigma,
        data = dat.for.plot)

px1 = seq(min(dat.for.plot$x1), max(dat.for.plot$x1), by =0.01)
px2 = seq(min(dat.for.plot$x2), max(dat.for.plot$x2), by =0.01)
grid_set = expand.grid(px1, px2)
colnames(grid_set) = c("x1","x2")

y_grid = predict(m, newdata = grid_set)
  
plot(dat.for.plot$x1,dat.for.plot$x2, 
     main = 'SVM', 
     xlab = "x1", ylab = 'x2', 
     xlim = range(px1), ylim = range(px2)) 
contour(px1, px2, matrix(as.numeric(y_grid), length(px1), length(px2)))
  points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'coral1', 'aquamarine'))
  points(dat.for.plot, pch = 21, bg = ifelse(dat.for.plot$y == 1, 'red2','blue3')) 
}

fig5 = SVM.gus(100,0.1)

## 1.2.3 Example Dataset 3 ##
dat = readMat("Desktop/programing/R/Coursera machine learning/Homework/machine-learning-ex6/ex6/ex6data3.mat")
dat.for.plot = data.frame(x1 = dat$X[,1], 
                          x2 = dat$X[,2],
                          y = dat$y)
fig6 = ggplot(dat.for.plot,aes(x = x1, y = x2, color = as.factor(y)))+
  geom_point()


outcome = tune.svm(y~., data =dat.for.plot, kernel = "radial",
                gamma = c(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30),cost = c(0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30))

fig7 = SVM.gus(outcome$best.parameters)

### 2 Spam Classication ###
## 2.1 Preprocessing Emails ##
fig8 <- read_csv("Desktop/programing/R/Coursera machine learning/Homework/machine-learning-ex6/ex6/emailSample1.txt", 
                         col_names = FALSE)

library(tm)
library(SnowballC)
library(stringr)
fig9 = fig8 %>%
  tolower() %>%
  gsub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
       "httpaddr", .) %>%
  gsub('\\S+@\\S+', 'emailaddr"',.) %>%
  gsub("[^[:alnum:]]", " ", .) %>%
  stemDocument() %>%
  str_split(., " ") %>%
  unlist()

fig10 = read_delim("Desktop/programing/R/Coursera machine learning/Homework/machine-learning-ex6/ex6/vocab.txt", 
            "\t", escape_double = FALSE, col_names = FALSE,  trim_ws = TRUE)
fig11 = which(fig10$X2 %in% fig9)


## 2.2 Extracting Features from Emails ##
temp = fig10$X2 
temp[fig11] = 1
temp[-fig11] = 0
fig12 = matrix(as.numeric(temp),ncol = 1)
# non-zero entries
length(which(fig12==1))

## 2.3 Training SVM for Spam Classication ##
train.dat = readMat("Desktop/programing/R/Coursera machine learning/Homework/machine-learning-ex6/ex6/spamTrain.mat")
test.dat = readMat("Desktop/programing/R/Coursera machine learning/Homework/machine-learning-ex6/ex6/spamTest.mat")

m.train = svm(train.dat$y ~train.dat$X,cost=0.1,
        type = "C-classification", 
        kernel = "linear",scale = F)

m.test = svm(test.dat$y ~test.dat$X,cost=0.1,
              type = "C-classification", 
              kernel = "linear",scale = F)

table(predict(m.train), train.dat$y, dnn=c("Prediction", "Actual")) 
table(predict(m.test,newdata = test.dat$X), test.dat$y, dnn=c("Prediction", "Actual")) 

## 2.4 Top Predictors for Spam ##

fig13 = c("our click remov guarante visit basenumb dollar will price pleas nbsp
most lo ga dollarnumb") %>%  
  str_split(., " ") %>%
  unlist()
indice =  which(fig10$X2 %in% fig13)
temp = fig10$X2 
temp[indice] = 1
temp[-indice] = 0
X= matrix(as.numeric(temp),ncol = 1)

is.spam = mean(c(predict(m.train,t(X))))-1 <0.5

