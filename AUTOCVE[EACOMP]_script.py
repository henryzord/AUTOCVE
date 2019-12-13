import argparse
import multiprocessing as mp
import os
import time

from functools import reduce
import itertools as it

import numpy as np
import pandas as pd
from AUTOCVE.AUTOCVE import AUTOCVEClassifier
from sklearn.preprocessing import Imputer
from weka.core import jvm
from weka.core.converters import Loader
from weka.core.dataset import Instances

from pbil.evaluations import evaluate_on_test, EDAEvaluation, collapse_metrics
from pbil.utils import parse_open_ml, create_metadata_path
import javabridge

GRACE_PERIOD = 0  # 60


def unweighted_area_under_roc(score_handler, some_arg, y_true):
    score = AUTOCVEClassifier.get_unweighted_area_under_roc(
        y_true=np.array(y_true, copy=False, order='C', dtype=np.int64),
        y_score=np.array(score_handler.y_scores, copy=False, order='C', dtype=np.float64)
    )
    return score


def get_evaluation(dataset_path, n_fold, train_probs, test_probs, seed, results_path, id_exp, id_trial):
    """

    :param dataset_path:
    :param n_fold:
    :return: A tuple (train-data, test_data), where each object is an Instances object
    """
    dataset_name = dataset_path.split('/')[-1]

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
    collapsed.to_csv(os.path.join(
        results_path, id_exp,
        'test_sample-%02.d_fold-%02.d.csv' % (id_trial, n_fold)), index=True
    )

    return test_pevaluation


def fit_predict_proba(estimator, X, y, X_test):
    if estimator is None:
        return None
    try:
        if np.any(np.isnan(X.values)) or np.any(np.isnan(X_test.values)):
            imputer=Imputer(strategy="median")
            imputer.fit(X)
            X=imputer.transform(X)
            X_test=imputer.transform(X_test)
        else:
            X=X.values  # TPOT operators need numpy format for been applied
            X_test=X_test.values
            y=y.values

        estimator.fit(X,y)
        return estimator.predict_proba(X_test)
    except Exception as e:
        return None


def execute_exp(
        id_trial, n_fold, datasets_path, d_id, n_generations, time_per_task, pool_size,
        mutation_rate_pool, crossover_rate_pool, n_ensembles, mutation_rate_ensemble, crossover_rate_ensemble,
        n_jobs, results_path, context, max_heap_size='2G', seed=None, subsample=1):

    jvm.start(max_heap_size=max_heap_size)

    p = AUTOCVEClassifier(
        generations=n_generations,
        population_size_components=pool_size,
        mutation_rate_components=mutation_rate_pool,
        crossover_rate_components=crossover_rate_pool,
        population_size_ensemble=n_ensembles,
        mutation_rate_ensemble=mutation_rate_ensemble,
        crossover_rate_ensemble=crossover_rate_ensemble,
        grammar='grammarPBIL',
        max_pipeline_time_secs=60,
        max_evolution_time_secs=time_per_task,
        n_jobs=n_jobs,
        random_state=seed,
        scoring=unweighted_area_under_roc,
        verbose=1
    )

    X_train, X_test, y_train, y_test, df_types = parse_open_ml(
        datasets_path=datasets_path, d_id=d_id, n_fold=n_fold
    )

    p.optimize(X_train, y_train, subsample_data=subsample)

    train_probs = fit_predict_proba(p.get_best_pipeline(), X_train, y_train, X_train).astype(np.float64)
    test_probs = fit_predict_proba(p.get_best_pipeline(), X_train, y_train, X_test).astype(np.float64)

    get_evaluation(
        dataset_path=os.path.join(datasets_path, d_id), n_fold=n_fold,
        seed=seed, train_probs=train_probs, test_probs=test_probs,
        results_path=results_path, id_exp=d_id, id_trial=id_trial
    )

    jvm.stop()


def __get_running_processes__(jobs, datasets_status, results_path, total_experiments, total_folds):
    if len(jobs) > 0:
        jobs[0].join()

        for job in jobs:  # type: mp.Process
            if not job.is_alive():
                jobs.remove(job)

    for dataset, finished in datasets_status.items():
        if not finished:
            if len(os.listdir(os.path.join(results_path, dataset))) == (total_experiments * total_folds):
                try:
                    summary = collapse_metrics(os.path.join(results_path, dataset), only_baselines=True)
                    datasets_status[dataset] = True
                    print('summary for dataset %s:' % dataset)
                    print(summary)
                except:
                    pass

    return jobs, datasets_status


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

    jvm.start()
    results_path = create_metadata_path(args=some_args)
    os.chdir(results_path)

    mp.set_start_method('spawn')
    context = mp.get_context('spawn')

    datasets_status = {k: False for k in datasets_names}

    queue_experiments = it.product(datasets_names, list(range(1, N_TRIALS + 1)), list(range(1, 10 + 1)))

    jobs = []
    for id_exp, id_trial, n_fold in queue_experiments:
        print("Dataset %s, trial %d, fold %d" % (id_exp, id_trial, n_fold))
        if not os.path.exists(os.path.join(results_path, id_exp)):
            os.mkdir(os.path.join(results_path, id_exp))

        if len(jobs) >= some_args.n_jobs:
            jobs, datasets_status = __get_running_processes__(
                jobs=jobs, datasets_status=datasets_status,
                results_path=results_path,
                total_experiments=N_TRIALS, total_folds=10,
            )

        job = mp.Process(
            target=execute_exp,
            kwargs=dict(
                n_fold=n_fold,
                id_trial=id_trial,
                datasets_path=some_args.datasets_path,
                d_id=id_exp,
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
                results_path=results_path,
                context=context,
                max_heap_size=some_args.heap_size
            )
        )
        job.start()
        jobs += [job]
        time.sleep(60)

    # blocks everything
    for job in jobs:
        job.join()

    # finishes everything
    __get_running_processes__(
        jobs=jobs, datasets_status=datasets_status,
        results_path=results_path,
        total_experiments=N_TRIALS, total_folds=10,
    )

    jvm.stop()


if __name__ == '__main__':
    main()
