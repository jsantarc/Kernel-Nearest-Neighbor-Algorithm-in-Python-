from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np

class KernelKNNClassifier:
    
   def __init__(self, Kernel,Parameters):
     self.Kernel = Kernel
     self.Parameters = Parameters
     self.KernelSelf=[];
     self.Features=[]; 
     self.Lables=[]; 
     self.Predicted=[]; 
     
     
   def fit(self,y,X):
          
       #Ipuput 
       #Train Kernel Nearest-Neighbor the distance is calculated by the following formula 
       #Distance=k(x,x)+k(y,y)+k(x,y)
       #this Method calculates k(x,x) and dose some housekeeping by putting info into the object 
       #self: object that Contains kernel name and parameter 
       #y:vector of labels
       #X: vector of features with columns  corresponding to sample and rows corresponding to features    
       #example        
       #Knn=KernelKNNClassifier(['sigmoid'],[1,1])
       #X = np.array([[1, 1], [0, 1],[1, 0], [-1, -1], [1, -1],[-1, 0],[100,100]]);
       #y=[1,1,1,0,0,0,2]
       #Knn.fit(y,X)
       #yhat=Y.predicte(X,1)
     y=np.array(y)
    #Number of training samples
     N=len(X[:,1]);
    #Array sorting value of kernels k(x,x)
     Kxx=np.zeros(N);
    #Calculate Kernel vector between same vectors Kxx[n]=k(X[n,:],X[n,:])
    #dummy for kernel name  
     Type=self.Kernel;
     for n in range(0,N):  
         
        Kxx[n]=pairwise_kernels(X[n,:], metric=Type[0],filter_params=self.Parameters)

    #Vector of kernel with the feature vectors  K(Xn,Xn) 
     self.KernelSelf=Kxx;
     #Features used for calculations    
     self.Features=X; 
     #Labels of  of the vectors the correspond to rows of X 
     self.Lables=y; 
  
   def predicte(self,Z,NaborsNumber):   
      #Z: matrix of to vectors to be classified  
    Nz=len(Z[:,1])
    #Empty list of predictions   
    yhat= np.zeros(Nz);
    #number of samples for classification
    Nz=len(Z[:,1]);
    #Number of training samples
    Nx=len(self.Features);
    #Dummy variable  Vector of ones used to get rid of one loop for k(z,z) 
    Ones=np.ones(Nx);
    
    #squared Euclidean distance in kernel space for each training sample
    Distance=np.zeros(Nx)
    # Index of sorted values 
    Index= np.zeros(Nx)
    
    Type=self.Kernel; 
     # calculate pairwise kernels beteen Training samples and prediction samples   
    Kxz=pairwise_kernels(self.Features,Z, metric=Type[0],filter_params=self.Parameters)
    
    NaborsNumberAdress=range(0,NaborsNumber)

    #Calculate Kernel vector between same vectors Kxx[n]=k(Z[n,:],Z[n,:]) 
     	
    for n in range(0,Nz):
        # calculate squared Euclidean distance in kernel space for each training sample 
        #for one prediction
        #for m in range(0,Nx)
        #Distance[m]=|phi(x[m])-phi(z[n])|^2=k(x,x)+k(z,z)-2k(z,x)    
                
        Distance =self.KernelSelf+pairwise_kernels(Z[n,:], metric=Type[0],filter_params=self.Parameters)*Ones-2*Kxz[:,n]   
    
        #Distance indexes sorted from smallest to largest  
        Index=np.argsort(Distance.flatten());
        Index=Index.astype(int)
        
        #get the labels of the nearest feature vectors          
        yl=list(self.Lables[Index[NaborsNumberAdress]])
        #perform majority vote 
        yhat[n]=max(set(yl), key=yl.count)
   
    return(yhat)