from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 



def load_data():  
    data = load_iris (as_frame = True)
    A= data.data 
    b=data.target 
    name_flower= data.target_names
    return A , b , name_flower



def split_data(A , b , test_size=0.3 , random_state=42 ): 
    return train_test_split(A , b  , test_size=test_size , random_state=random_state , stratify=b)
if __name__ == "__main__": 
    A , b , name_flower = load_data()
    A_train , A_test , b_train , b_test = split_data (A , b)
    print("Train shape :" , A_train.shape)
    print("Test shape :" , A_test.shape)