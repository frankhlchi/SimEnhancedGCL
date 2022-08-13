import numpy as np
import functools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


@repeat(3)
def label_classification(embeddings, y, train_mask, valid_mask, test_mask, ratio):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')

    X_train = X[train_mask.detach().cpu().numpy(),:]
    X_valid = X[valid_mask.detach().cpu().numpy(),:]
    X_test= X[test_mask.detach().cpu().numpy(),:]
    
    y_train = Y[train_mask.detach().cpu().numpy(),:]
    y_valid = Y[valid_mask.detach().cpu().numpy(),:]
    y_test = Y[test_mask.detach().cpu().numpy(),:]
    
    print ( 'X_train: ',X_train.shape)                                              
    print ( 'X_valid: ', X_valid.shape)
    print ( 'X_train: ',X_test.shape)
    
    logreg = LogisticRegression(solver='liblinear', random_state =0)
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)
    
    y_pred_test = clf.predict_proba(X_test)
    y_pred_test = prob_to_one_hot(y_pred_test)
    micro_test = f1_score(y_test, y_pred_test, average="micro")
    macro_test = f1_score(y_test, y_pred_test, average="macro")

    y_pred_valid = clf.predict_proba(X_valid)
    y_pred_valid = prob_to_one_hot(y_pred_valid)
    micro_valid = f1_score(y_valid, y_pred_valid, average="micro")
    macro_valid = f1_score(y_valid, y_pred_valid, average="macro")

    return {
        'valid': micro_valid,
        'test': micro_test
    }
