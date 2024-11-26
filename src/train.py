import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
#import joblib

if __name__ == '__main__':

    # Load train set
    train_dataset = pd.read_csv('data/external/train.csv')

    # Get X and Y
    y = train_dataset.loc[:, 'target'].values.astype("float32")
    X = train_dataset.drop('target', axis=1).values

    # Create an instance of Logistic Regression Classifier and fit the data.
    clf = LogisticRegression(C=0.01, solver='lbfgs', multi_class='multinomial', max_iter=100)
    clf.fit(X, y)

    with open('models/model.pkl','wb') as f:
        pickle.dump(clf, f)

    #joblib.dump(clf, 'models/model.joblib')
