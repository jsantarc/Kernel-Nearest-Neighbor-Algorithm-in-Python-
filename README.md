#Kernel-Nearest-Neighbor-Algorithm-in-Python-
This code extended to the well-known nearest-neighbor algorithm for classification so that kernels can be used 

the distance is calculated by the following formula 
Distance=k(x,x)+k(y,y)-2k(x,y)


#Object
obj=KernelKNNClassifier(Kernel)
Kernel:name of kernal 
Parameters: kernal parameters in not parameters just put 0
see kernel type of more info 

#Methos
fit(y,X)
y:vector of labels
X: vector of features with columns corresponding to sample and rows corresponding to features    
predicte(self,Z,NaborsNumber)
Z:Feature matrix of vectors to be predicted. Rows correspond to sample index  
NaborsNumber: number of nabors for Majority vote 



#Example 
Knn=KernelKNNClassifier(['sigmoid'],[1,1])
X = np.array([[1, 1], [0, 1],[1, 0], [-1, -1], [1, -1],[-1, 0],[100,100]]);
y=[1,1,1,0,0,0,2]
Knn.fit(y,X)
two Nabors 
Nabors=2
yhat=Knn.predicte(X, Nabors)

#Kernel references  
1)Type:'rbf'
Parameters=[gamma] 
Formula:K(x, y) = exp(-gamma ||x-y||^2)  
 
2)Type:'sigmoid'
Parameters=[ gamma , c0]
Formula:K(x, y) = tanh(gamma  <X, Y>+c0)
gamma : slope
c0 is known as intercep

3)Type:‘polynomial’
Parameters=[gamma,coef0,degree] 
Formula:K(X, Y) = (gamma <X, Y> + coef0)^degree
coef0 : int, default 1
degree : int, default 3
3)Type:‘lin’
Formula:K(X, Y) =  <X, Y> 
  
4)Type:‘cosine’
L2-normalized dot product of vectors.
K(X, Y) =  <X, Y>/||X||||Y||
