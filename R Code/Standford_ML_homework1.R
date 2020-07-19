#### Simple Octave/MATLAB function ####
diag(5)

#### Linear regression with one variable ####
library(readr)
data = read.delim("ex1data1.txt",header = F,sep = ",")
x = data[,1]
y = data[,2]
m = length(y)
library(ggplot2)
ggplot(data,aes(x = x,y= y))+
  geom_point()+
  title("MarkerSize")



### gradient descent 
gradientR<-function(y, X, epsilon,eta, iters){
  epsilon = 0.0001
  X = as.matrix(data.frame(rep(1,length(y)),X))
  N= dim(X)[1]
  print("Initialize parameters...")
  theta.init = as.matrix(rnorm(n=dim(X)[2], mean=0,sd = 1)) # Initialize theta
  theta.init = t(theta.init)
  e = t(y) - theta.init%*%t(X)
  grad.init = -(2/N)%*%(e)%*%X
  theta = theta.init - eta*(1/N)*grad.init
  print(theta)
  l2loss = c()
  theta.iter = c()
  for(i in 1:iters){
    l2loss = c(l2loss,sqrt(sum((t(y) - theta%*%t(X))^2)))
    e = t(y) - theta%*%t(X)
    grad = -(2/N)%*%e%*%X
    theta = theta - eta*(2/N)*grad
    theta.iter = rbind(theta.iter,c(theta[,1],theta[,2],i))
    if(sqrt(sum(grad^2)) <= epsilon){
      break
    }
  }
  print("Algorithm converged")
  print(paste("Final gradient norm is",sqrt(sum(grad^2))))
  values<-list("coef" = t(theta), "l2loss" = l2loss)
  return(values)
}


gdec.eta1 = gradientR(y = y, X = x, eta = 0.01, iters = 1500)

plot(1:length(gdec.eta1$l2loss),gdec.eta1$l2loss,xlab = "Epoch", ylab = "L2-loss")
lines(1:length(gdec.eta1$l2loss),gdec.eta1$l2loss)

final.theta = gdec.eta1$coef

predict1 = c(1,3.5) %*% final.theta
predict2 = c(1,7) %*% final.theta


ggplot(data,aes(x = x,y= y))+
  geom_point()+
  title("MarkerSize")+
  geom_abline(intercept = final.theta[1,] , slope = final.theta[2,],colour = "red")

library(plotly)
gradientR.theta<-function(y, X, epsilon,eta, iters){
  epsilon = 0.0001
  X = as.matrix(data.frame(rep(1,length(y)),X))
  N= dim(X)[1]
  print("Initialize parameters...")
  theta.init = as.matrix(rnorm(n=dim(X)[2], mean=0,sd = 1)) # Initialize theta
  theta.init = t(theta.init)
  e = t(y) - theta.init%*%t(X)
  grad.init = -(2/N)%*%(e)%*%X
  theta = theta.init - eta*(1/N)*grad.init
  print(theta)
  l2loss = c()
  theta.iter = c()
  for(i in 1:iters){
    l2loss = c(l2loss,sqrt(sum((t(y) - theta%*%t(X))^2)))
    e = t(y) - theta%*%t(X)
    grad = -(2/N)%*%e%*%X
    theta = theta - eta*(2/N)*grad
    theta.iter = rbind(theta.iter,c(theta[,1],theta[,2]))
    if(sqrt(sum(grad^2)) <= epsilon){
      break
    }
  }
  print("Algorithm converged")
  print(paste("Final gradient norm is",sqrt(sum(grad^2))))
  values<-list("coef" = t(theta), "l2loss" = l2loss)
  return(theta.iter)
}

theta.layout = gradientR.theta(y = y, X = x, eta = 0.01, iters = 1500)



f<-function(u,v){
  u*u*exp(2*v)+4*v*v*exp(-2*u)-4*u*v*exp(v-u)
}

x = sort(theta.layout[,1])
y = sort(theta.layout[,2])
z <- outer(x,y,f)
#Contour plot
contour(x,y,z)
#Persp plot
persp(x, y, z, phi = 25, theta = 55, 
      xlim=c(0,1),
      ylim=c(0.5,2),
      xlab = "theta0", ylab = "theta1",
      main = "", col="yellow", ticktype = "detailed"
) -> res


#### Linear regression with multiple variables ####
library(readr)
library(dplyr)

data2 = read.delim("ex1data2.txt",header = F,sep = ",")
piv = data2 %>%
  summarise(mean1 = mean(V1),
            mean2 = mean(V2),
            mean3 = mean(V3),
            sd1 = sd(V1),
            sd2 = sd(V2),
            sd3 = sd(V3))

X = data.frame((data2[,1]-2000)/794.7,(data2[,2]-3.17)/0.76) #with normalization
y = (data2[,3]-340412)/125039.9 #with normalization
eta = c(0.3,0.1,0.03,0.01)
gdec.eta3 = gradientR(y = y, X = X, eta = 2, iters = 50)
plot(1:length(gdec.eta3$l2loss),gdec.eta3$l2loss,xlab = "Epoch", ylab = "L2-loss",color = "green")
lines(1:length(gdec.eta3$l2loss),gdec.eta3$l2loss)
## after the regression processing, y-value scaling back 

#### 3.3 Normal Equations ####
X = data.frame(data2[,1],data2[,2]) 
y = data2[,3]
X  = as.matrix(data.frame(rep(1,length(y)),X))
solve(t(X) %*% X) %*% t(X) %*% y


# test the normal equation with normalization
X = data.frame((data2[,1]-2000)/794.7,(data2[,2]-3.17)/0.76) #with normalization
y = (data2[,3]-340412)/125039.9 
X  = as.matrix(data.frame(rep(1,length(y)),X))
solve(t(X) %*% X) %*% t(X) %*% y
