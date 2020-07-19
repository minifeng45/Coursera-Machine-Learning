library(readr)
#####1####
data1 = read.delim("Desktop/machine-learning-ex2/ex2/ex2data1.txt",sep=",")

### 1.1 Visualizing the data 
library(ggplot2)
pass.or.not = as.factor(data1[,3])
ggplot(data1, aes(x = data1[,1], y= data1[,2],color = pass.or.not)) +
  geom_point()


###1.2 Implementation####
library(e1071)
sigmoid(data1)

## Cost function and gradient
#Predictor variables
X <- as.matrix(data1[,c(1,2)])

#Add ones to X
X <- cbind(rep(1,nrow(X)),X)

#Response variable
Y <- as.matrix(data1[,3])

#Cost Function
cost <- function(theta)
{
  m <- nrow(X)
  g <- sigmoid(X%*%theta)
  J <- (1/m)*sum((-Y*log(g)) - ((1-Y)*log(1-g)))
  return(J)
}

#Intial theta
initial_theta <- rep(0,ncol(X))

#Cost at inital theta
cost(initial_theta)

# Derive theta using gradient descent using optim function
theta_optim <- optim(par=initial_theta,fn=cost)

#set theta
theta <- theta_optim$par

#cost at optimal value of the theta
theta_optim$value

# probability of admission for student
prob <- sigmoid(t(c(1,67.31926,66.58935))%*%theta)

pass.or.not = as.factor(data1[,3])
ggplot(data1, aes(x = data1[,1], y= data1[,2],color = pass.or.not)) +
  geom_point()+
  geom_line(aes(x = data1[,1], y =122.285538-0.98288034*data1[,1] ,color = "blue"))


#####2####
##2.1 Visualizing the data
data2 = read.delim("Desktop/machine-learning-ex2/ex2/ex2data2.txt",sep=",")
pass.or.not = as.factor(data2[,3])
ggplot(data2, aes(x = data2[,1], y= data2[,2],color = pass.or.not)) +
  geom_point()

##2.2 Feature mapping
x1 = data2[,1]
x2 = data2[,2]
X = list()
degree = 6 
k=1
for ( i in 1:degree){
  for(j in 0:i){
    X[[k]] = x1^(i-j)*x2^j
    k = k+1
  }
}

X.mat = c(rep(1,length(x1)))
for (k  in 1:27){
  X.mat = cbind(X.mat,X[[k]])
}
X = X.mat
##2.3 Cost function and gradient

#Response variable
Y <- as.matrix(data2[,3])

#Cost Function
cost <- function(theta)
{
  m <- nrow(X)
  g <- sigmoid(X%*%theta)
  J <- (1/m)*sum((-Y*log(g)) - ((1-Y)*log(1-g)))
  return(J)
}

#Intial theta
initial_theta <- rep(0,ncol(X.mat))

#Cost at inital theta
cost(initial_theta)

# Derive theta using gradient descent using optim function
theta_optim <- optim(par=initial_theta,fn=cost)

#set theta
theta <- theta_optim$par

#cost at optimal value of the theta
theta_optim$value


ggplot(data2, aes(x = data2[,1], y= data2[,2],color = pass.or.not)) +
  geom_point()+
  geom_contour(aes(x=x1, y=x2))
