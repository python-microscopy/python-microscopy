def pearson(X, Y):
    X = X - X.mean()
    Y = Y-Y.mean()
    return (X*Y).sum()/sqrt((X*X).sum()*(Y*Y).sum())

def overlap(X, Y):
    return (X*Y).sum()/sqrt((X*X).sum()*(Y*Y).sum())