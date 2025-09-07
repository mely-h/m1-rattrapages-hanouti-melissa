from prepare import load_data , split_data 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt 



def evaluate():
 A , b , _  = load_data ()
 A_train , A_test , b_train , b_test = split_data(A ,b )



 logistic = LogisticRegression(max_iter=150)
 logistic.fit(A_train , b_train)
 prediction_logistic = logistic.predict(A_test)
 precision_logistic = accuracy_score(b_test , prediction_logistic)
 print("Logistic :" , precision_logistic) 




 knearest = KNeighborsClassifier(n_neighbors=4)
 knearest.fit(A_train , b_train)
 prediction_knearst = knearest.predict(A_test)
 precision_knearest = accuracy_score(b_test , prediction_knearst)
 print("Knearest :" , precision_knearest) 


 #comparaison 
 if precision_logistic > precision_knearest :
    print("Logistic est plus precis que Knearest")
 else :
    print("Knearest est plus precis que Logistic")  
    

 plt.figure(figsize=(6 , 3 ))
 plt.bar(["Logistic" , "Knearest"],[precision_logistic , precision_knearest] , color=["black", "red"] ) 
 plt.ylabel("Precisions")
 plt.title("Comparaison des modeles")
 plt.ylim(0,1)
 plt.show()
    
if __name__ == "__main__": 
        evaluate()    