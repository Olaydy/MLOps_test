import json
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import pickle
#import joblib


if __name__ == '__main__':
    
    classes = pd.read_csv('data/raw/iris.csv')['target'].unique().tolist()
    
    test_dataset = pd.read_csv('data/external/test.csv')
    y = test_dataset.loc[:, 'target'].values.astype("float32")
    X = test_dataset.drop('target', axis=1).values
    
    #clf = joblib.load('data/model.joblib')
    with open('models/model.pkl', 'rb') as f:
        trained_clf = pickle.load(f)
    
    prediction = trained_clf.predict(X)
    cm = confusion_matrix(prediction, y)
    f1 = f1_score(y_true=y, y_pred=prediction, average='macro')

    json.dump(
        obj={
            'f1_score': f1,
            'confusion_matrix': {
                'classes': classes,
                'matrix': cm.tolist()
            }
        },
        fp=open('reports/eval.txt', 'w')
    )
