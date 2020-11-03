import argparse
import os
from functools import reduce

import javabridge
import numpy as np
import pandas as pd
from AUTOCVE.AUTOCVE import AUTOCVEClassifier
from AUTOCVE.util.evaluate import unweighted_area_under_roc
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from weka.core import jvm
from weka.core.converters import Loader
from weka.core.dataset import Instances

from pbil.evaluations import evaluate_on_test, EDAEvaluation
from pbil.utils import parse_open_ml, create_metadata_path

GRACE_PERIOD = 0  # 60


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


def execute_exp(
        n_sample, n_fold, datasets_path, dataset_name, n_generations, time_per_task, pool_size,
        mutation_rate_pool, crossover_rate_pool, n_ensembles, mutation_rate_ensemble, crossover_rate_ensemble,
        n_jobs, results_path, seed=None, subsample=1):

    this_experiment_path = os.path.join(results_path, dataset_name, 'sample_%02.d_fold_%02.d' % (n_sample, n_fold))
    os.mkdir(this_experiment_path)
    os.chdir(this_experiment_path)

    p = AUTOCVEClassifier(
        generations=n_generations,
        population_size_components=pool_size,
        mutation_rate_components=mutation_rate_pool,
        crossover_rate_components=crossover_rate_pool,
        population_size_ensemble=n_ensembles,
        mutation_rate_ensemble=mutation_rate_ensemble,
        crossover_rate_ensemble=crossover_rate_ensemble,
        grammar='grammarPBIL',  # TODO reactivate
        # grammar='grammarTPOT',
        max_pipeline_time_secs=60,  # TODO review
        max_evolution_time_secs=time_per_task,  # TODO review
        n_jobs=n_jobs,
        random_state=seed,
        scoring=unweighted_area_under_roc,  # function was reviewed and is operating as intended
        verbose=1
    )

    X_train, X_test, y_train, y_test, df_types = parse_open_ml(
        datasets_path=datasets_path, d_id=dataset_name, n_fold=n_fold
    )

    p.optimize(X_train, y_train, subsample_data=subsample, n_classes=len(np.unique(y_train)))

    train_probs = fit_predict_proba(p.get_best_pipeline(), X_train, y_train, X_train).astype(np.float64)
    test_probs = fit_predict_proba(p.get_best_pipeline(), X_train, y_train, X_test).astype(np.float64)

    get_evaluation(
        dataset_path=os.path.join(datasets_path, dataset_name), n_fold=n_fold,
        seed=seed, train_probs=train_probs, test_probs=test_probs,
        results_path=os.path.join(results_path, dataset_name, 'overall'), id_trial=n_sample
    )


def main():
    parser = argparse.ArgumentParser(
        description='Script for running Helio and Celio algorithm, with hyper-parameters from EACOMP.'
    )

    parser.add_argument(
        '--experiment-description', action='store', required=False, default="",
        help='Description of experiment'
    )

    parser.add_argument(
        '--n-samples', action='store', required=True, type=int,
        help='Description of experiment'
    )

    parser.add_argument(
        '--metadata-path', action='store', required=True,
        help='Path to put results of experiments'
    )

    parser.add_argument(
        '--datasets-path', action='store', required=True,
        help='Path where datasets are.'
    )

    parser.add_argument(
        '--datasets-names', action='store', required=True,
        help='Name of the datasets to run experiments. Must be a list separated by a comma '
             'Example:\n'
             'python script.py --datasets-names iris,mushroom,adult'
    )

    parser.add_argument(
        '--n-jobs', action='store', required=False, default=-1, type=int,
        help='Number of parallel threads to run. Defaults to -1 (i.e use all cores)'
    )

    parser.add_argument(
        '--n-generations', action='store', required=True, type=int,
        help='Number of generations to run'
    )

    parser.add_argument(
        '--time-per-task', action='store', required=False, type=int, default=8000,
        help='Maximum time (in seconds) that a single run of the algorithm is allowed to run'
    )

    parser.add_argument(
        '--pool-size', action='store', required=False, type=int, default=50,
        help='Maximum number of classifiers to have in the pool (i.e. pre-ensemble) at any given time'
    )

    parser.add_argument(
        '--mutation-rate-pool', action='store', required=False, type=float, default=0.9,
        help='Mutation rate for classifiers in the pool'
    )

    parser.add_argument(
        '--crossover-rate-pool', action='store', required=False, type=float, default=0.9,
        help='Crossover rate for classifiers in the pool'
    )

    parser.add_argument(
        '--n-ensembles', action='store', required=False, type=int, default=50,
        help='Maximum number of ensembles to have at any given time'
    )

    parser.add_argument(
        '--mutation-rate-ensemble', action='store', required=False, type=float, default=0.1,
        help='Mutation rate for ensembles'
    )

    parser.add_argument(
        '--crossover-rate-ensemble', action='store', required=False, type=float, default=0.9,
        help='Crossover rate for ensembles'
    )

    parser.add_argument(
        '--heap-size', action='store', required=False, default='2G',
        help='string that specifies the maximum size, in bytes, of the memory allocation pool. '
             'This value must be a multiple of 1024 greater than 2MB. Append the letter k or K to indicate kilobytes, '
             'or m or M to indicate megabytes. Defaults to 2G'
    )

    some_args = parser.parse_args()

    if 0 < some_args.n_samples <= 20:
        N_TRIALS = some_args.n_samples
    else:
        raise Exception("The number of trials is expected to be an integer with value between 1 and 20.")

    datasets_names = some_args.datasets_names.split(',')

    jvm.start(max_heap_size=some_args.heap_size)
    results_path = create_metadata_path(args=some_args)

    # datasets_status = {k: False for k in datasets_names}

    # queue_experiments = it.product(datasets_names, list(range(1, N_TRIALS + 1)), list(range(1, 10 + 1)))

    try:
        execute_exp(
            n_sample=1,
            n_fold=1,
            datasets_path=some_args.datasets_path,
            dataset_name=datasets_names[0],
            seed=np.random.randint(np.iinfo(np.int32).max),
            n_generations=some_args.n_generations,
            n_jobs=1,
            time_per_task=some_args.time_per_task,
            pool_size=some_args.pool_size,
            mutation_rate_pool=some_args.mutation_rate_pool,
            crossover_rate_pool=some_args.crossover_rate_pool,
            n_ensembles=some_args.n_ensembles,
            mutation_rate_ensemble=some_args.mutation_rate_ensemble,
            crossover_rate_ensemble=some_args.crossover_rate_ensemble,
            results_path=results_path
        )
        jvm.stop()
    except Exception as e:
        jvm.stop()
        raise e


if __name__ == '__main__':
    main()
