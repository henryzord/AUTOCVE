from collections import Counter

from weka.core import jvm
from AUTOCVE.pbil.generation import J48
import numpy as np
from pbil.utils import parse_open_ml

from AUTOCVE.AUTOCVE import get_unweighted_area_under_roc


def evaluate_pipeline(pipeline, X_train, y_train, X_test):

    pipeline.fit(X_train, y_train)
    predict_data = []

    if getattr(pipeline, 'predict_proba', None) is not None:
        predict_scores = pipeline.predict_proba(X_test)
        predict_data = np.argmax(predict_scores, axis=1).astype(np.int64)
    else:
        predict_data = pipeline.predict(X_test).astype(np.int64)
        n_classes = len(Counter(y_train))
        predict_scores = np.zeros((len(X_test), n_classes), dtype=np.float64)
        indices = np.arange(len(X_test))
        predict_scores[indices, predict_data.astype(np.int64)] = 1.

    return predict_data, predict_scores


def main():
    jvm.start()
    try:
        X_train, X_test, y_train, y_test, df_types = parse_open_ml(
            datasets_path='C:\\Users\\henry\\Projects\\ednel\\keel_datasets_10fcv',
            d_id='iris',
            n_fold=1
        )
        # X_train.columns = [i for i, x in enumerate(X_train.columns)]
        # y_train.name = '4'
        # X_test.columns = X_train.columns

        clf = J48()
        predict_data, predict_scores = evaluate_pipeline(clf, X_train, y_train, X_test)
        print(predict_scores)

        # clf.fit(X_train.values, y_train.values)
        # scores = clf.predict_proba(X_test.values)
        # print(get_unweighted_area_under_roc(
        #     y_true=y_test.values, y_score=np.array(scores, copy=True, order='C', dtype=np.float64)
        # ))
        jvm.stop()
    except Exception as e:
        jvm.stop()
        raise e


def compare_fitness():
    # define 5 arrays of y_trues and y_scores for test
    y_true = np.array([0, 0, 1, 1, 2, 2], copy=False, order='C', dtype=np.int64)

    print('truth array:\n', y_true)

    a_array = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float64)
    b_array = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    c_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    t1 = np.array(np.vstack((a_array, b_array, c_array)).T, copy=True, order='C', dtype=np.float64)
    t2 = np.array(np.vstack((a_array, c_array, b_array)).T, copy=True, order='C', dtype=np.float64)
    t3 = np.array(np.vstack((c_array, b_array, a_array)).T, copy=True, order='C', dtype=np.float64)
    t4 = np.array(np.vstack((c_array, a_array, b_array)).T, copy=True, order='C', dtype=np.float64)
    t5 = np.array(np.vstack((b_array, a_array, c_array)).T, copy=True, order='C', dtype=np.float64)
    t6 = np.array(np.vstack((b_array, c_array, a_array)).T, copy=True, order='C', dtype=np.float64)

    tests = [t1, t2, t3, t4, t5, t6]
    for test in tests:
        print(test)
        print(get_unweighted_area_under_roc(y_true=y_true, y_score=test))


if __name__ == '__main__':
    main()
