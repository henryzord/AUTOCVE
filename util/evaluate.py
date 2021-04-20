import os
import tempfile
import time
import warnings
from collections import Counter
from functools import partial
from multiprocessing import TimeoutError
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
from AUTOCVE.AUTOCVE import get_unweighted_area_under_roc
from joblib import delayed
from joblib import dump, load
from joblib.externals.loky.process_executor import TerminatedWorkerError
from sklearn.model_selection import StratifiedKFold, train_test_split

from .joblib_silent_timeout import ParallelSilentTimeout
from .make_pipeline import make_pipeline_str

y_last_test_set = None


def log_warning_output(message, category, filename, lineno, file=None, line=None):
    with open("log_warning_methods.log", "a+") as file:
        file.write(
            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ": " +
            warnings.formatwarning(message, category, filename, lineno) + "\n"
        )


warnings.showwarning = log_warning_output


class ScorerHandler(object):
    def __init__(self, y_pred, y_scores):
        """

        :param y_pred: The predicted labels for instances
        :param y_scores:  A matrix where each row is an instance and each column the class probabilities, given by
            the classifier.
        """

        self.y_pred = y_pred
        self.y_scores = y_scores

    def fit(self):
        pass

    def predict(self, X):
        return self.y_pred

    def predict_proba(self, X):
        return self.y_scores


def evaluate_population_holdout(pipelines_population, X, y, scoring, n_jobs, timeout_pip_sec, N_SPLITS=5, verbose=1, RANDOM_STATE=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=RANDOM_STATE)

    try:
        global y_last_test_set
        y_last_test_set = y_test

        # list_train_index = []
        # list_test_index = []
        # y_last_test_set = []

        # split = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE)

        # for train_index, test_index in split.split(X, y):
        #     list_train_index.append(train_index)
        #     list_test_index.append(test_index)
        #     if len(y_last_test_set) == 0:
        #         y_last_test_set = y[test_index]
        #     else:
        #         y_last_test_set = np.concatenate([y_last_test_set, y[test_index]])

        # predict_length = y.shape[0]
        predict_length = y_test.shape[0]

        pipelines_population = pipelines_population.split("|")

        temp_folder = tempfile.mkdtemp()
        filename_train = os.path.join(temp_folder, 'autocve_X_train.mmap')
        filename_test = os.path.join(temp_folder, 'autocve_X_test.mmap')

        metric_population = []
        predict_population = []
        evaluate_pipe_timeout = partial(evaluate_solution, verbose=verbose)

        # for train_index, test_index in zip(list_train_index, list_test_index):

        if os.path.exists(filename_train):
            os.unlink(filename_train)
        if os.path.exists(filename_test):
            os.unlink(filename_test)
        _ = dump(X_train, filename_train)
        _ = dump(X_test, filename_test)

        X_train = load(filename_train, mmap_mode='r')
        X_test = load(filename_test, mmap_mode='r')

        result_pipeline = ParallelSilentTimeout(n_jobs=n_jobs, backend="loky", timeout=timeout_pip_sec)(
            delayed(evaluate_pipe_timeout)(pipeline_str, X_train, X_test, y_train, y_test) for
            pipeline_str in pipelines_population if pipeline_str is not None)

        metric_population_cv = []
        predict_population_cv = []

        next_pipe = iter(result_pipeline)
        for pipe_id, pipe_str in enumerate(pipelines_population):
            if pipe_str is None:
                metric_population_cv.append(None)
                predict_population_cv.append(None)
            else:
                result_solution = next(next_pipe)
                if isinstance(result_solution, TimeoutError):
                    if verbose > 0:
                        print("Timeout reach for pipeline: " + str(pipe_str))
                    metric_population_cv.append(None)
                    predict_population_cv.append(None)
                    pipelines_population[pipe_id] = None
                elif isinstance(result_solution, TerminatedWorkerError):
                    if verbose > 0:
                        print("Worker error for pipeline: " + str(pipe_str))
                    metric_population_cv.append(None)
                    predict_population_cv.append(None)
                    pipelines_population[pipe_id] = None
                elif result_solution is None:
                    metric_population_cv.append(None)
                    predict_population_cv.append(None)
                    pipelines_population[pipe_id] = None
                else:
                    metric_population_cv.append([scoring(ScorerHandler(*result_solution), y_test)])
                    predict_population_cv.append(result_solution[0])

        del result_pipeline

        if len(metric_population) == 0:
            metric_population = metric_population_cv
            predict_population = predict_population_cv
        else:
            metric_population = [None if value is None or old_list is None else old_list + value for old_list, value
                                 in zip(metric_population, metric_population_cv)]
            predict_population = [None if value is None or old_list is None else np.concatenate([old_list, value])
                                  for old_list, value in zip(predict_population, predict_population_cv)]

        metric_population = [None if metrics is None else np.mean(metrics) for metrics in metric_population]

        # raises PermissionError on windows
        try:
            os.unlink(filename_train)
            os.unlink(filename_test)
            os.rmdir(temp_folder)
        except PermissionError:
            pass  # leaves for OS do deal with it later

        return metric_population, predict_population, predict_length

    except (KeyboardInterrupt, SystemExit) as e:
        try:
            os.unlink(filename_train)
            os.unlink(filename_test)
            os.rmdir(temp_folder)
        except Exception as e:
            pass

        return None, None, -1


def evaluate_population_cv(pipelines_population, X, y, scoring, n_jobs, timeout_pip_sec, N_SPLITS=5, verbose=1, RANDOM_STATE=42):
    try:
        global y_last_test_set
        y_last_test_set = None

        list_train_index = []
        list_test_index = []
        y_last_test_set = []

        split = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE)

        for train_index, test_index in split.split(X, y):
            list_train_index.append(train_index)
            list_test_index.append(test_index)
            if len(y_last_test_set) == 0:
                y_last_test_set = y[test_index]
            else:
                y_last_test_set = np.concatenate([y_last_test_set, y[test_index]])

        predict_length = y.shape[0]

        pipelines_population = pipelines_population.split("|")

        temp_folder = tempfile.mkdtemp()
        filename_train = os.path.join(temp_folder, 'autocve_X_train.mmap')
        filename_test = os.path.join(temp_folder, 'autocve_X_test.mmap')

        metric_population = []
        predict_population = []
        evaluate_pipe_timeout = partial(evaluate_solution, verbose=verbose)

        for train_index, test_index in zip(list_train_index, list_test_index):
            X_train = X[train_index, :]
            X_test = X[test_index, :]
            if os.path.exists(filename_train):
                os.unlink(filename_train)
            if os.path.exists(filename_test):
                os.unlink(filename_test)
            _ = dump(X_train, filename_train)
            _ = dump(X_test, filename_test)

            X_train = load(filename_train, mmap_mode='r')
            X_test = load(filename_test, mmap_mode='r')

            result_pipeline = ParallelSilentTimeout(n_jobs=n_jobs, backend="loky", timeout=timeout_pip_sec)(
                delayed(evaluate_pipe_timeout)(pipeline_str, X_train, X_test, y[train_index], y[test_index]) for
                pipeline_str in pipelines_population if pipeline_str is not None)

            metric_population_cv = []
            predict_population_cv = []

            next_pipe = iter(result_pipeline)
            for pipe_id, pipe_str in enumerate(pipelines_population):
                if pipe_str is None:
                    metric_population_cv.append(None)
                    predict_population_cv.append(None)
                else:
                    result_solution = next(next_pipe)
                    if isinstance(result_solution, TimeoutError):
                        if verbose > 0:
                            print("Timeout reach for pipeline: " + str(pipe_str))
                        metric_population_cv.append(None)
                        predict_population_cv.append(None)
                        pipelines_population[pipe_id] = None
                    elif isinstance(result_solution, TerminatedWorkerError):
                        if verbose > 0:
                            print("Worker error for pipeline: " + str(pipe_str))
                        metric_population_cv.append(None)
                        predict_population_cv.append(None)
                        pipelines_population[pipe_id] = None
                    elif result_solution is None:
                        metric_population_cv.append(None)
                        predict_population_cv.append(None)
                        pipelines_population[pipe_id] = None
                    else:
                        metric_population_cv.append([scoring(ScorerHandler(*result_solution), y[test_index])])
                        predict_population_cv.append(result_solution[0])

            del result_pipeline

            if len(metric_population) == 0:
                metric_population = metric_population_cv
                predict_population = predict_population_cv
            else:
                metric_population = [None if value is None or old_list is None else old_list + value for old_list, value
                                     in zip(metric_population, metric_population_cv)]
                predict_population = [None if value is None or old_list is None else np.concatenate([old_list, value])
                                      for old_list, value in zip(predict_population, predict_population_cv)]

        metric_population = [None if metrics is None else np.mean(metrics) for metrics in metric_population]

        # raises PermissionError on windows
        try:
            os.unlink(filename_train)
            os.unlink(filename_test)
            os.rmdir(temp_folder)
        except PermissionError:
            pass  # leaves for OS do deal with it later

        return metric_population, predict_population, predict_length

    except (KeyboardInterrupt, SystemExit) as e:
        try:
            os.unlink(filename_train)
            os.unlink(filename_test)
            os.rmdir(temp_folder)
        except Exception as e:
            pass

        return None, None, -1


def evaluate_population(pipelines_population, X, y, scoring, n_jobs, timeout_pip_sec, N_SPLITS=5, verbose=1, RANDOM_STATE=42):
    if N_SPLITS > 0:
        return evaluate_population_cv(pipelines_population, X, y, scoring, n_jobs, timeout_pip_sec, N_SPLITS, verbose, RANDOM_STATE)
    return evaluate_population_holdout(pipelines_population, X, y, scoring, n_jobs, timeout_pip_sec, N_SPLITS, verbose, RANDOM_STATE)



def evaluate_solution(
        pipeline_str, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, verbose=1
):
    pipeline = make_pipeline_str(pipeline_str, verbose)
    if pipeline is None:
        return None
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        if verbose > 0:
            print("Pipeline fit error: " + str(e) + "\n")
            print(str(pipeline))
        return None

    try:
        if getattr(pipeline, 'predict_proba', None) is not None:
            predict_scores = pipeline.predict_proba(X_test)
            predict_data = np.argmax(predict_scores, axis=1).astype(np.int64)
        else:
            predict_data = pipeline.predict(X_test).astype(np.int64)
            n_classes = len(Counter(y_train))
            predict_scores = np.zeros((len(X_test), n_classes), dtype=np.float64)
            indices = np.arange(len(X_test))
            predict_scores[indices, predict_data.astype(np.int64)] = 1.

    except Exception as e:
        if verbose > 0:
            print("Pipeline predict error: " + str(e) + "\n")
            print(str(pipeline))
        return None

    return predict_data, predict_scores


# scoring is the scoring function passed as argument to AUTOCVEClassifier
# in this case, should be function unweighted_area_under_roc
def evaluate_predict_vector(predict_vector, predict_scores, scoring):
    return scoring(ScorerHandler(predict_vector, predict_scores), y_last_test_set)


# one valid scoring function
def unweighted_area_under_roc(score_handler, y_true):
    # TODO changed from C++ code to sklearn code!
    scores = score_handler.y_scores
    n_classes = scores.shape[1]

    auc = 0.
    for c in range(n_classes):
        actual_binary_class = (y_true == c).astype(np.int)  # TODO check if classes are linear!!!!
        auc += roc_auc_score(y_true=actual_binary_class, y_score=scores[:, c])

    return auc / n_classes

    # score = get_unweighted_area_under_roc(
    #     y_true=np.array(y_true, copy=False, order='C', dtype=np.int64),
    #     y_score=np.array(score_handler.y_scores, copy=False, order='C', dtype=np.float64)
    # )
    # return score
