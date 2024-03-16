import numpy as np
import os 
os.system("cls")


class Node: # new  data type
    def _init_(self , feature = None ,value = None,left = None ,right = None, result = None   ):  
         # for decision node  node we want to know which feature and which value we will split , its right and left
       
        self.feature = feature
        self.value = value 
        self.left = left 
        self.right = right

        # for leaf node we need to know the output or its prediction 
        self.result = result 



class DecisionTree:
    def _init_(self , max_depth = 3 ,criterion = "entropy"):
        self.max_depth = max_depth
        self.tree= None
        self.criterion = criterion


    def split_data (x , feature , value ):     # function to split data take data,feature and the value to split 
        # assume that the decisin node    (x <= 12 )   
        left_indices = np.where(x[:,feature]<=value)[0]  # if true 
        right_indices = np.where(x[:,feature]>=value)[0] # if false
        return left_indices , right_indices
    
    def entropy (self ,y ) : # function to calculate entropy (y >> data )
        p = np.bincount (y)/len(y)  # get the probability for  unique values (y)
        return -np.sum (p*np.log2(p+1e-10))
    
    def gini(self,y):
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p**2)
    
    def infoemation_gain( self , y , left_indices , right_indices):
        if self.criterion == "entropy":
            impurity_function = self.entropy
        elif self.criterion == "gini":
            impurity_function = self.gini
        parent_impurity = impurity_function(y)
        left_impurity = impurity_function(y[left_indices])
        right_impurity = impurity_function(y[right_indices])

        w_left = len(left_indices) / len(y)
        w_right = len(right_indices) / len(y)

        information_gain = parent_impurity - w_left * left_impurity - w_right * right_impurity
        return information_gain
    
    def fit (self , x, y  , depth = 0):     # which split is better ?
        if depth == self.max_depth or np.all(y[0] == y):   # to stop
            return Node(result = y[0]) 
        samples , features = x.shape
        best_information_gain = 0.0 
        best_split = None 

        for i in range (features):     # feature 
            feature_value = x[:,i]
            unique_values = np.unique (feature_value)
            for j in unique_values :     # value 
                left_indices , rigth_indices = self.split(x,i,j)
                if len(left_indices) == 0 or len(right_indices) ==0:
                    continue 
                information_gain = self.infoemation_gain(y,left_indices,rigth_indices)
                if information_gain < best_information_gain :
                    best_information_gain = information_gain  
                    best_split = (i,j ,left_indices , rigth_indices)
        if best_information_gain == 0.0 :  # leaf node 
            return Node(result = np.bincount(y).argmax)   # maximum unique values to classify
        
        feature , value , left_indices , rigth_indices = best_split
        left_subtree = self.fit(x[left_indices,y[left_indices]],depth+1)
        rigth_subtree = self.fit(x[rigth_indices] , y[rigth_indices],depth+1)
        self.tree = Node (feature = feature , value = value ,left = left_subtree , right = rigth_subtree)
        return self.tree
    


    def predict (self , x ) :

         result = [self.predict_recursive(point,self.tree) for point in X]
         return np.array(result)



    def predict_recursive(self, point , node):
        if node.result is not None:
            return node.result
        
        if point[node.feature] <= node.value:
            return self.predict_recursive(point, node.left)
        else:
            return self.predict_recursive(point, node.right)





 
 
