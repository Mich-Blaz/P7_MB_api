# -*- coding: utf-8 -*-
"""
@author: MichBlaz
"""

# On importe les libraries necessaire poyue le bon fonctionnement du code
import uvicorn
from fastapi import FastAPI
from Clientdatas import Clientdata
import numpy as np
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
# Creation de l app et import des choses necessaores
app = FastAPI()

@app.get('/Welcome')
def get_name(name: str):
    return {'Welcome to an API for prediction of loan giving ': f'{name}'}






print(type(app))
# loader du classifier et de l'explainer

clf = pickle.load(open("Pipeline_final_clf.pkl","rb"))


print(str(clf))



filename_expl = 'explainer_final.sav'
load_explainer = pickle.load(open(filename_expl, 'rb'))


print(str(load_explainer))

df_nn=pd.read_csv('d_sample_to_api.csv')
print(df_nn.head())

preproc=clf['preprocessor']
model=clf['classifier']

neigh = NearestNeighbors(n_neighbors=30)
neigh.fit(preproc.transform(df_nn.drop('TARGET',axis=1)))

null=np.nan

@app.post('/predict_val_shap_neigh')
def predict_credit(data:Clientdata):
    df_client=pd.DataFrame(data.dict(),index=[0])

    xclient=preproc.transform(df_client)

    idx = neigh.kneighbors(X=xclient,n_neighbors=20,return_distance=False).ravel()
    nn=df_nn.iloc[idx]
    nn=nn.replace({np.nan:None})

    prediction_proba = model.predict_proba(xclient)
    pred = model.predict(xclient)

    shapval = dict(pd.Series(load_explainer.shap_values(xclient)[0][0],index=df_client.columns))


    return {'pred_class': pred[0],'pred_proba':prediction_proba.max(),'shap_val':shapval, 'Nearest_client':nn.to_dict() }



if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload

