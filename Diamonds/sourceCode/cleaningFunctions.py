from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


cut_ranks = {
    "Fair": 0,
    "Good": 1,
    "Very Good": 2,
    "Premium": 3,
    "Ideal": 4
}
color_ranks = {
    'D': 7,
    'E': 6,
    'F': 5,
    'G': 4,
    'H': 3,
    'I': 2,
    'J': 1}

clarity_ranks = {'I1': 0, ' SI2': 1, ' SI1': 2, ' VS2': 3,
                 ' VS1': 4, ' VVS2': 5, ' VVS1': 6, ' IF': 7}


class CleanDiamonds(TransformerMixin):
    def __init__(self, df):
        self.features = ['id', 'carat', 'cut', 'color',
                         'clarity', 'depth', 'table', 'x', 'y', 'z']
        self.df = df
        self.X = self.df[self.features]

    # def fit_ordinal(self, X):
    #     # Get only interesting data
    #     X = df[self.features]
    #     # Replace cut with integers (ranked from worst to best)
    #     X['cut'] = X['cut'].map(cut_nums)
    #     # Replace color with with integers (ranked from worst to best)
    #     X['color'] = X['color'].map(color_ranks)
    #     # Replace clarity with integers (ranked from worst to best)
    #     X['clarity'] = X['clarity'].map(clarity_ranks)
    #     return self

    def fit(self):
        X = self.X
        # Replace cut with integers (ranked from worst to best)
        X['cut'] = X['cut'].map(cut_ranks)
        # Replace color with with integers (ranked from worst to best)
        X['color'] = X['color'].map(color_ranks)
        # Replace clarity with integers (ranked from worst to best)
        X['clarity'] = X['clarity'].map(clarity_ranks)
        
        for e in ['cut', 'color', 'clarity']:
            X[e] = X[e].map(str)
            X = pd.get_dummies(X)
        scaler = MinMaxScaler()
        X[['carat', 'depth', 'x', 'y', 'z']] = pd.DataFrame(
            columns=['carat', 'depth', 'x', 'y', 'z'],
            data=scaler.fit_transform(X[['carat', 'depth', 'x', 'y', 'z']]))
        return self

    # def minMaxScaling(self, X):
    #     X = df[self.features]
    #     scaler = MinMaxScaler()
    #     X[['carat', 'depth', 'x', 'y', 'z']] = pd.DataFrame(
    #         columns=['carat', 'depth', 'x', 'y', 'z'],
    #         data=scaler.fit_transform(X[['carat', 'depth', 'x', 'y', 'z']]))
    #     return self

    def transform(self):
        return self.X
