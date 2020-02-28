from sklearn.base import TransformerMixin

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

class CleanTitanic(TransformerMixin):
    def __init__(self):
        self.features = ['id', 'carat', 'cut', 'color',
                         'clarity', 'depth', 'table', 'x', 'y', 'z']

    def fit(self, X):
        # Get only interesting data
        X = df[self.features]
        # Replace cut with integers (ranked from worst to best)
        X['cut'] = X['cut'].map(cut_nums)
        # Replace color with with integers (ranked from worst to best)
        X['color'] = X['color'].map(color_ranks)
        # Replace clarity with integers (ranked from worst to best)
        X['clarity'] = X['clarity'].map(clarity_ranks)
        return self

    def transform(self, df):
        return self.X
