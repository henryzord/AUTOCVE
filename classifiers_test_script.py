from weka.core import jvm
from AUTOCVE.pbil.generation import SimpleCart
import numpy as np

from AUTOCVE.AUTOCVE import get_unweighted_area_under_roc


def main():
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
