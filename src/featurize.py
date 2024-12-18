import pandas as pd

def get_features(dataset):

    features = dataset.copy()
    # Rename columns: replace (cm) and spaces
    features.rename(
        columns=lambda s: s.replace('(cm)', '').strip().replace(' ', '_'),
        inplace=True
    )

    features['sepal_length_to_sepal_width'] = features['sepal_length'] / features['sepal_width']
    features['petal_length_to_petal_width'] = features['petal_length'] / features['petal_width']

    return features


if __name__ == '__main__':
    dataset = pd.read_csv('data/raw/iris.csv')

    features  = get_features(dataset)
    features.to_csv('data/interim/features_iris.csv', index=False)
