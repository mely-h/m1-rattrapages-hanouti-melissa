from sklearn.linear_model import LogisticRegression 
from prepare import load_data , split_data
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt 


def train_evaluate(): 
    A , b , name_flower = load_data() 
    A_train , A_test , b_train , b_test = split_data(A , b) 
    
    
    modele = LogisticRegression(max_iter=150)
    modele.fit(A_train , b_train)
    
    prediction=modele.predict(A_test)
    
    precision=accuracy_score(b_test , prediction)
    print("régression logistique:" , precision)
    
    plt.figure(figsize=(3 , 3 ))
    plt.bar(["Logistic"], [precision] , color=["black"] ) 
    plt.ylabel("Precisions")
    plt.title("Précision du modèle Logistic")
    plt.ylim(0,1)
    plt.show()
    
if __name__ == "__main__": 
        train_evaluate()
    
