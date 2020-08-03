#Let's start with importing necessary libraries
import pickle
#import numpy as np
import pandas as pd

class predObj:

    def predict_log(self, dict_pred):
        with open("standardScalar.pkl", 'rb') as f:
            scalar = pickle.load(f)
            predmodel = dict_pred.get("Model")
            print(predmodel)
            del dict_pred['Model']
            print(dict_pred)
        if predmodel == "Logistic Regression":
            with open("Logistic_ModelForPrediction.pkl", 'rb') as f:
                model = pickle.load(f)
            data_df = pd.DataFrame(dict_pred,index=[1,])
            scaled_data = scalar.transform(data_df)
            predict = model.predict(scaled_data)
            accuracy = 0.75
            f1 = 0.74
            precision = 0.74
            recall = 0.75

        elif predmodel == "SVM":
            with open("SVM_ModelForPrediction.pkl", 'rb') as f:
                model = pickle.load(f)
            data_df = pd.DataFrame(dict_pred,index=[1,])
            scaled_data = scalar.transform(data_df)
            predict = model.predict(scaled_data)
            accuracy = 0.79
            f1 = 0.79
            precision = 0.78
            recall = 0.79
        else :
            with open("RandomForest_ModelForPrediction.pkl", 'rb') as f:
                model = pickle.load(f)
            data_df = pd.DataFrame(dict_pred,index=[1,])
            scaled_data = scalar.transform(data_df)
            predict = model.predict(scaled_data)
            accuracy = 0.86
            f1 = 0.86
            precision = 0.86
            recall = 0.85

        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'

        return (result, accuracy, f1, precision, recall, predmodel)