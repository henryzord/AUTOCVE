import json
import time
from copy import deepcopy
from multiprocessing import set_start_method

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from mPBIL.utils import path_to_dataframe

try:
    set_start_method("spawn")
except RuntimeError:
    pass  # is in child process, trying to set context to spawn but failing because is already set

import multiprocessing as mp
from datetime import datetime as dt
import argparse
import logging
import os
import sys
from functools import reduce, wraps

import javabridge
from weka.core.classes import Random
import numpy as np
import pandas as pd
from AUTOCVE.AUTOCVE import AUTOCVEClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from weka.core import jvm
from weka.core.converters import Loader
from weka.core.dataset import Instances

from pbil.evaluations import evaluate_on_test, EDAEvaluation
from util.evaluate import unweighted_area_under_roc

GRACE_PERIOD = 0  # 60


def parse_open_ml(datasets_path: str, d_id: str, n_fold: int):
    """
    Function that processes each dataset into an interpretable form

    Warning: will convert categorical attributes to one-hot encoding.
    """
    # X_train, X_test, y_train, y_test, df_types
    train = path_to_dataframe('{0}-10-{1}tra.arff'.format(os.path.join(datasets_path, str(d_id), str(d_id)), n_fold))
    test = path_to_dataframe('{0}-10-{1}tst.arff'.format(os.path.join(datasets_path, str(d_id), str(d_id)), n_fold))

    df_types = pd.DataFrame(
        dict(name=train.columns, type=['categorical' if str(x) == 'category' else 'numerical' for x in train.dtypes]))
    df_types.loc[df_types['name'] == df_types.iloc[-1]['name'], 'type'] = 'target'

    categorical_columns = []
    dict_convs = []

    for i, column in enumerate(train.columns):
        if str(train[column].dtype) == 'category' or (i == len(train.columns) - 1):
            categories = train[column].dtype.categories
            dict_conv = dict(zip(categories, range(len(categories))))
            train.loc[:, column] = train.loc[:, column].replace(dict_conv).astype(np.int32)

            dict_convs += [dict_conv]
            categorical_columns += [column]

    for column, dict_conv in zip(categorical_columns, dict_convs):
        test.loc[:, column] = test.loc[:, column].replace(dict_conv).astype(np.int32)

    X_train = train[train.columns[:-1]]
    y_train = train[train.columns[-1]]
    X_test = test[test.columns[:-1]]
    y_test = test[test.columns[-1]]

    return X_train.values, X_test.values, y_train.values, y_test.values, df_types


def get_evaluation(dataset_path, n_fold, train_probs, test_probs, seed, results_path, id_trial):
    """

    :param dataset_path:
    :param n_fold:
    :return: A tuple (train-data, test_data), where each object is an Instances object
    """
    dataset_name = dataset_path.split(os.sep)[-1]

    train_path = os.path.join(dataset_path, '-'.join([dataset_name, '10', '%dtra.arff' % n_fold]))
    test_path = os.path.join(dataset_path, '-'.join([dataset_name, '10', '%dtst.arff' % n_fold]))

    loader = Loader("weka.core.converters.ArffLoader")
    train_data = loader.load_file(train_path)
    train_data.class_is_last()

    test_data = loader.load_file(test_path)
    test_data.class_is_last()

    filter_obj = javabridge.make_instance('Lweka/filters/unsupervised/instance/Randomize;', '()V')
    javabridge.call(filter_obj, 'setRandomSeed', '(I)V', 1)
    javabridge.call(filter_obj, 'setInputFormat', '(Lweka/core/Instances;)Z', train_data.jobject)
    jtrain_data = javabridge.static_call(
        'Lweka/filters/Filter;', 'useFilter',
        '(Lweka/core/Instances;Lweka/filters/Filter;)Lweka/core/Instances;',
        train_data.jobject, filter_obj
    )
    jtest_data = javabridge.static_call(
        'Lweka/filters/Filter;', 'useFilter',
        '(Lweka/core/Instances;Lweka/filters/Filter;)Lweka/core/Instances;',
        test_data.jobject, filter_obj
    )

    train_data = Instances(jtrain_data)
    test_data = Instances(jtest_data)

    env = javabridge.get_env()  # type: javabridge.JB_Env
    jtrain_probs = env.make_object_array(train_probs.shape[0], env.find_class('[D'))  # type: numpy.ndarray
    for i in range(train_probs.shape[0]):
        row = env.make_double_array(np.ascontiguousarray(train_probs[i, :]))
        env.set_object_array_element(jtrain_probs, i, row)

    jtest_probs = env.make_object_array(test_probs.shape[0], env.find_class('[D'))
    for i in range(test_probs.shape[0]):
        row = env.make_double_array(np.ascontiguousarray(test_probs[i, :]))
        env.set_object_array_element(jtest_probs, i, row)

    clf = javabridge.make_instance(
        'Leda/NonJavaClassifier;',
        '([[D[[DLweka/core/Instances;Lweka/core/Instances;)V',
        jtrain_probs,
        jtest_probs,
        jtrain_data,
        jtest_data
    )

    test_evaluation_obj = evaluate_on_test(jobject=clf, test_data=test_data)
    test_pevaluation = EDAEvaluation.from_jobject(test_evaluation_obj, data=test_data, seed=seed)

    dict_metrics = dict()
    dict_models = dict()
    for metric_name, metric_aggregator in EDAEvaluation.metrics:
        value = getattr(test_pevaluation, metric_name)
        if isinstance(value, np.ndarray):
            new_value = np.array2string(value.ravel().astype(np.int32), separator=',')
            new_value_a = 'np.array(%s, dtype=np.%s).reshape(%s)' % (new_value, value.dtype, value.shape)
            value = new_value_a

        dict_metrics[metric_name] = value

    df_metrics = pd.DataFrame(dict_metrics, index=['AUTOCVE'])
    dict_models['AUTOCVE'] = df_metrics

    collapsed = reduce(lambda x, y: x.append(y), dict_models.values())
    collapsed.to_csv(
        os.path.join(results_path, 'test_sample-%02.d_fold-%02.d.csv' % (id_trial, n_fold)
        ), index=True
    )

    return test_pevaluation


def fit_predict_proba(estimator: Pipeline, X: np.ndarray, y: np.ndarray, X_test: np.ndarray):
    if estimator is None:
        return None
    try:
        if np.any(np.isnan(X)) or np.any(np.isnan(X_test)):
            imputer = SimpleImputer(strategy="median")
            imputer.fit(X)
            X = imputer.transform(X)
            X_test = imputer.transform(X_test)
        else:
            X = X
            X_test = X_test
            y = y

        estimator.fit(X, y)
        return estimator.predict_proba(X_test)
    except Exception as e:
        return None


def process_wrapper(func):
    @wraps(func)
    def inner(kwargs):
        return func(**kwargs)

    return inner


def run_external_fold(
        n_external_fold: int, n_internal_folds: int,
        dataset_name: str, datasets_path: str,
        metadata_path: str, experiment_folder: str
):
    some_exception = None  # type: Exception

    try:
        logger = init_logger(os.path.join(metadata_path, experiment_folder), n_external_fold)

        logger.info("Starting: Dataset %s, external fold %d" % (dataset_name, n_external_fold))

        random_state = 1
        seed = Random(random_state)

        logger.info('reading datasets')

        ext_train_X, ext_test_X, ext_train_y, ext_test_y, df_types = parse_open_ml(
            datasets_path=datasets_path, d_id=dataset_name, n_fold=n_external_fold
        )

        class_unique_values = sorted(np.unique(ext_train_y))

        logger.info('loading combinations')

        combinations = get_autocve_combinations()  # type: list

        logger.info('initializing class')

        aucs = []  # type: list

        for comb in combinations:
            preds = []
            internal_actual_classes = []

            stratifier = StratifiedKFold(n_splits=n_internal_folds, random_state=random_state)

            for n_internal_fold, (train_index, test_index) in enumerate(stratifier.split(ext_train_X, ext_train_y)):
                int_train_X = ext_train_X[train_index]
                int_train_y = ext_train_y[train_index]

                int_test_X = ext_train_X[test_index]
                int_test_y = ext_train_y[test_index]

                # internal_actual_classes.extend(list(internal_test_data.values(internal_test_data.class_index)))

                autocve = AUTOCVEClassifier(
                    generations=comb['generations'],
                    population_size_components=comb['population_size_components'],
                    mutation_rate_components=comb['mutation_rate_components'],
                    crossover_rate_components=comb['crossover_rate_components'],
                    population_size_ensemble=comb['population_size_ensemble'],
                    mutation_rate_ensemble=comb['mutation_rate_ensemble'],
                    crossover_rate_ensemble=comb['crossover_rate_ensemble'],
                    grammar=comb['grammar'],
                    max_pipeline_time_secs=comb['max_pipeline_time_secs'],
                    max_evolution_time_secs=comb['max_evolution_time_secs'],
                    n_jobs=comb['n_jobs'],
                    random_state=comb['random_state'],
                    scoring=comb['scoring'],
                    verbose=comb['verbose'],
                    cv_folds=comb['cv_folds']
                )

                logger.info('building classifier')
                autocve.optimize(
                    int_train_X, int_train_y,
                    subsample_data=1,  # TODO unknown effect!!!!
                    n_classes=len(class_unique_values)
                )

                logger.info('getting best individual')

                best_ensemble = autocve.get_best_voting_ensemble()  # type: VotingClassifier
                best_ensemble.voting = 'soft'
                best_ensemble.fit(int_train_X, int_train_y)

                preds.extend(list(map(list, best_ensemble.predict_proba(int_test_X))))
                internal_actual_classes.extend(list(int_test_y))

            internal_actual_classes = np.array(internal_actual_classes, dtype=np.int)
            preds = np.array(preds)

            auc = 0.
            for i, c in enumerate(class_unique_values):
                actual_binary_class = (internal_actual_classes == i).astype(np.int)
                auc += roc_auc_score(y_true=actual_binary_class, y_score=preds[:, i])

            aucs += [auc / len(class_unique_values)]

        best_index = int(np.argmax(aucs))  # type: int
        best_combination = combinations[best_index]

        autocve = AUTOCVEClassifier(
            generations=best_combination['generations'],
            population_size_components=best_combination['population_size_components'],
            mutation_rate_components=best_combination['mutation_rate_components'],
            crossover_rate_components=best_combination['crossover_rate_components'],
            population_size_ensemble=best_combination['population_size_ensemble'],
            mutation_rate_ensemble=best_combination['mutation_rate_ensemble'],
            crossover_rate_ensemble=best_combination['crossover_rate_ensemble'],
            grammar=best_combination['grammar'],
            max_pipeline_time_secs=best_combination['max_pipeline_time_secs'],
            max_evolution_time_secs=best_combination['max_evolution_time_secs'],
            n_jobs=best_combination['n_jobs'],
            random_state=best_combination['random_state'],
            scoring=best_combination['scoring'],
            verbose=best_combination['verbose'],
            cv_folds=best_combination['cv_folds']
        )

        autocve.optimize(
            ext_train_X, ext_train_y,
            subsample_data=1,  # TODO unknown effect!!!!
            n_classes=len(class_unique_values)
        )

        clf = autocve.get_best_voting_ensemble()  # type: VotingClassifier
        clf.voting = 'soft'
        clf.fit(ext_train_X, ext_train_y)

        external_preds = list(map(list, clf.predict_proba(ext_test_X)))
        external_actual_classes = list(ext_test_y)

        with open(
                os.path.join(metadata_path, experiment_folder, dataset_name,
                             'test_sample-01_fold-%02d_parameters.json' % n_external_fold),
                'w'
        ) as write_file:
            json.dump(combinations[best_index], write_file, indent=2)

        with open(
                os.path.join(metadata_path, experiment_folder, dataset_name, 'overall',
                             'test_sample-01_fold-%02d_overall.preds' % n_external_fold)
                , 'w') as write_file:
            write_file.write('classValue;Individual\n')
            for i in range(len(external_actual_classes)):
                write_file.write('%r;%s\n' % (external_actual_classes[i], ','.join(map(str, external_preds[i]))))

    except Exception as e:
        some_exception = e
    finally:
        if some_exception is not None:
            logger.error('Finished with exception set:', str(some_exception))
#             print(some_exception.args[0], file=sys.stderr)
            raise some_exception
        else:
            logger.info("Finished: Dataset %s, external fold %d" % (dataset_name, n_external_fold))


def get_autocve_combinations():
    combinations = []

    print('TODO to implement:')
    print('fixar tamanho do vetor de bits do GA em 5 (usar uma população do GP diferente = 50)')
    print('aggregation function? implementar no auto-cve nossas funções de agregação')
    time.sleep(2)

    generations = 50  # TODO change to 100!
    population_size_components = 50  # at most 5 classifiers for each ensemble
    population_size_ensemble = 50  # same value used by EDNEL
    grammar = 'grammarPBILlight'  # grammar without data transformations
    # grammar = 'grammarTPOT'  # grammar to be used with interpretable models
    max_evolution_time_secs = 3600  # same value used by EDNEL
    max_pipeline_time_secs = 60  # same value used by EDNEL
    random_state = 1
    n_jobs = 1
    scoring = unweighted_area_under_roc  # function was reviewed and is operating as intended
    verbose = 0  # shut up!
    n_folds = 0

    # GP
    mutation_rate_components_array = [0.7, 0.9]
    crossover_rate_components_array = [0.7, 0.9]
    # GA
    mutation_rate_ensemble = 0.1  # fixo
    crossover_rate_ensemble_array = [0.7, 0.8, 0.9]

    for mutation_rate_components in mutation_rate_components_array:
        for crossover_rate_components in crossover_rate_components_array:
            for crossover_rate_ensemble in crossover_rate_ensemble_array:
                comb = {
                    'generations': generations,
                    'population_size_components': population_size_components,
                    'population_size_ensemble': population_size_ensemble,
                    'grammar': grammar,
                    'max_evolution_time_secs': max_evolution_time_secs,
                    'max_pipeline_time_secs': max_pipeline_time_secs,
                    'random_state': random_state,
                    'n_jobs': n_jobs,
                    'scoring': scoring,
                    'verbose': verbose,
                    'mutation_rate_components': mutation_rate_components,
                    'crossover_rate_components': crossover_rate_components,
                    'mutation_rate_ensemble': mutation_rate_ensemble,
                    'crossover_rate_ensemble': crossover_rate_ensemble,
                    'cv_folds': n_folds  # uses holdout
                }
                combinations += [comb]

    return combinations


def read_dataset(path: str) -> Instances:
    loader = Loader("weka.core.converters.ArffLoader")  # type: weka.core.converters.Loader

    data = loader.load_file(path)
    data.class_is_last()

    filter_obj = javabridge.make_instance('Lweka/filters/unsupervised/instance/Randomize;', '()V')
    javabridge.call(filter_obj, 'setRandomSeed', '(I)V', 1)
    javabridge.call(filter_obj, 'setInputFormat', '(Lweka/core/Instances;)Z', data.jobject)
    jtrain_data = javabridge.static_call(
        'Lweka/filters/Filter;', 'useFilter',
        '(Lweka/core/Instances;Lweka/filters/Filter;)Lweka/core/Instances;',
        data.jobject, filter_obj
    )
    data = Instances(jtrain_data)
    return data


def get_params(args: argparse.Namespace) -> dict:
    """
    Get parameters of script that is running. Makes a copy.

    :param args: The parameters as passed to this script
    :type args: argparse.Namespace
    :return: the parameters, as a dictionary
    :rtype: dict
    """
    return deepcopy(args.__dict__)


def create_metadata_folder(some_args: argparse.Namespace, metadata_path: str, dataset_name: str) -> str:
    experiment_folder = dt.now().strftime('%Y-%m-%d-%H-%M-%S')

    os.mkdir(os.path.join(metadata_path, experiment_folder))
    os.mkdir(os.path.join(metadata_path, experiment_folder, dataset_name))
    os.mkdir(os.path.join(metadata_path, experiment_folder, dataset_name, 'overall'))

    with open(os.path.join(metadata_path, experiment_folder, 'parameters.json'), 'w') as write_file:
        dict_params = get_params(some_args)
        json.dump(dict_params, write_file, indent=2)

    return experiment_folder


def init_logger(experiment_path, n_external_fold):
    # create logger with 'application'
    logger = logging.getLogger('application')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(experiment_path, 'external_fold_%02d_process_%d.log' % (n_external_fold, os.getpid())))
    fh.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)

    return logger


def start_jvms(heap_size, experiment_path):
    if not jvm.started:
        jvm.start(max_heap_size=heap_size)


def stop_jvms(_):
    if jvm.started:
        jvm.stop()


def main(args: argparse.Namespace):
    e = None

    n_jobs = args.n_jobs
    n_external_folds = 10  # TODO do not change this
    n_internal_folds = args.n_internal_folds

    experiment_folder = create_metadata_folder(args, args.metadata_path, args.dataset_name)

    os.chdir(os.path.join(args.metadata_path, experiment_folder))

    if n_jobs == 1:
        print('WARNING: using single-thread.')
        time.sleep(2)

        some_exception = None
        jvm.start(max_heap_size=args.heap_size)
        try:
            for i in range(1, n_external_folds + 1):
                run_external_fold(
                    i, n_internal_folds,
                    args.dataset_name, args.datasets_path,
                    args.metadata_path, experiment_folder
                )
        except Exception as e:
            some_exception = e
        finally:
            jvm.stop()
            if some_exception is not None:
                raise some_exception
    else:
        print('Using %d processes' % n_jobs)
        time.sleep(2)

        with mp.Pool(processes=n_jobs) as pool:
            iterable_params = [
                (x, n_internal_folds,
                 args.dataset_name, args.datasets_path,
                 args.metadata_path, experiment_folder
                 ) for x in range(1, n_external_folds + 1)]

            pool.map(start_jvms, iterable=[
                (args.heap_size, os.path.join(args.metadata_path, experiment_folder)
                 ) for x in range(1, n_external_folds + 1)
            ])
            pool.starmap(func=run_external_fold, iterable=iterable_params)
            pool.map(stop_jvms, iterable=range(1, n_external_folds + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for running nested cross-validation AUTOCVE'
    )

    parser.add_argument(
        '--heap-size', action='store', required=False, default='2G',
        help='string that specifies the maximum size, in bytes, of the memory allocation pool. '
             'This value must be a multiple of 1024 greater than 2MB. Append the letter k or K to indicate kilobytes, '
             'or m or M to indicate megabytes. Defaults to 2G'
    )

    parser.add_argument(
        '--metadata-path', action='store', required=True,
        help='Path to where all datasets are stored'
    )

    parser.add_argument(
        '--datasets-path', action='store', required=True,
        help='Path to where all datasets are stored'
    )

    parser.add_argument(
        '--dataset-name', action='store', required=True,
        help='Name of dataset to run nested cross validation'
    )

    parser.add_argument(
        '--n-internal-folds', action='store', required=True,
        help='Number of folds to use to perform an internal cross-validation for each combination of hyper-parameters',
        type=int,
        choices=set(range(1, 6))
    )

    parser.add_argument(
        '--n-jobs', action='store', required=False,
        help='Number of parallel threads to use when running this script',
        type=int, choices=set(range(1, 11)), default=1
    )

    main(args=parser.parse_args())
