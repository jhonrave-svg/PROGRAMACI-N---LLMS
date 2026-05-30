import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.pipeline import Pipeline

def analizar_resiliencia_hidraulica(X, y):

    pipeline = Pipeline([
        ('normalizer', Normalizer(norm='l2')),
        ('modelo', VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000, solver='liblinear')),
                ('kn', KNeighborsClassifier(n_neighbors=3))
            ],
            voting='hard'
        ))
    ])

    tscv = TimeSeriesSplit(n_splits=4)

    mcc_scorer = make_scorer(matthews_corrcoef)

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=tscv,
        scoring=mcc_scorer
    )

    return float(scores.mean())
