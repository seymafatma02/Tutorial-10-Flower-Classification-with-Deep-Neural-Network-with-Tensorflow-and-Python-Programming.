import os
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import pickle 
# cicek verisi inince ekle 



data_dir = 'C:\\pyton.py\\cicekverisi\\flowers'
categories=['daisy','rose','sunflower','tulip']

data= []

def make_data():
    for category in categories :
        path=os.path.join(data_dir,category) #./data/flower/daisy
        label= categories.index(category)
        
        for img_name in os.listdir(path):
            image_path=os.path.join(path, img_name)
            image = cv2.imread(image_path)
     
            try:
                image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image =cv2.resize(image,(224,224))

                image =np.array(image,dtype=np.float32)

                data.append([image,label])
            except Exception as e :
                print(f"Error proccesing image {image_path}:{e}")

                
    
    print(len(data))
    with open('data.pickle', 'wb') as pik:
    
       pickle.dump(data,pik)
   
            
   

make_data()

def load_data():
    with open('data.pickle', 'rb') as pick:
        data = pickle.load(pick)
    
    pick.close()

    np.random.shuffle(data)

    feature =[]
    labels=[]

    for img,label in data :
        feature.append(img)
        labels.append(label)

    feature =np.array(feature,dtype=np.float32)
    labels=np.array(labels)

    return [feature,labels]

