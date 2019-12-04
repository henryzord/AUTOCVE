import argparse
import json
import multiprocessing as mp
import os
from datetime import datetime as dt
from functools import reduce

import numpy as np
import pandas as pd
from AUTOCVE.AUTOCVE import AUTOCVEClassifier
from scipy.io import arff
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from weka.core import jvm
from weka.core.converters import Loader
from weka.core.dataset import Instances

from pbil.evaluations import evaluate_on_test, EDAEvaluation, collapse_metrics

GRACE_PERIOD = 0  # 60


def unweighted_area_under_roc(score_handler, some_arg, y_true):
    return AUTOCVEClassifier.get_unweighted_area_under_roc(
        y_true=y_true,
        y_score=score_handler.y_pred
    )


def create_metadata_path(args):
    now = dt.now()

    str_time = now.strftime('%d-%m-%Y-%H:%M:%S')

    joined = os.getcwd() if not os.path.isabs(args.metadata_path) else ''
    to_process = [args.metadata_path, str_time]

    for path in to_process:
        joined = os.path.join(joined, path)
        if not os.path.exists(joined):
            os.mkdir(joined)

    with open(os.path.join(joined, 'parameters.json'), 'w') as f:
        json.dump({k: getattr(args, k) for k in args.__dict__}, f, indent=2)

    return joined
    # these_paths = []
    # for dataset_name in datasets_names:
    #     local_joined = os.path.join(joined, dataset_name)
    #     these_paths += [local_joined]
    #
    #     if not os.path.exists(local_joined):
    #         os.mkdir(local_joined)
    #
    # return joined


def get_evaluation(dataset_path, n_fold, train_probs, test_probs, seed, results_path, id_exp, id_trial):
    """

    :param dataset_path:
    :param n_fold:
    :return: A tuple (train-data, test_data), where each object is an Instances object
    """
    import javabridge
    jvm.start()

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

    print('got collapsed everything')
    jvm.stop()

    return test_pevaluation


def path_to_dataframe(dataset_path):
    """
    Reads dataframes from an .arff file, casts categorical attributes to categorical type of pandas.

    :param dataset_path:
    :return:
    """

    value, metadata = path_to_arff(dataset_path)

    df = pd.DataFrame(value, columns=metadata._attrnames)

    attributes = metadata._attributes
    for attr_name, (attr_type, rang_vals) in attributes.items():
        if attr_type in ('nominal', 'string'):
            df[attr_name] = df[attr_name].apply(lambda x: x.decode('utf-8'))

            df[attr_name] = df[attr_name].astype('category')
        elif attr_type == 'date':
            raise TypeError('unsupported attribute type!')
        else:
            df[attr_name] = df[attr_name].astype(np.float32)

    return df


def path_to_arff(dataset_path):
    """
    Given a path to a dataset, reads and returns a dictionary which comprises an arff file.

    :type dataset_path: str
    :param dataset_path: Path to the dataset. Must contain the .arff file extension (i.e., "my_dataset.arff")
    :rtype: dict
    :return: a dictionary with the arff dataset.
    """

    dataset_type = dataset_path.split('.')[-1].strip()
    assert dataset_type == 'arff', TypeError('Invalid type for dataset! Must be an \'arff\' file!')
    af = arff.loadarff(dataset_path)
    return af


def parse_open_ml(datasets_path, d_id, n_fold, queue=None):
    """Function that processes each dataset into an interpretable form
    Args:
        d_id (int): dataset id
    Returns:
        A tuple of the train / test split data along with the column types
    """
    jvm.start()

    # X_train, X_test, y_train, y_test, df_types
    train = path_to_dataframe('{0}-10-{1}tra.arff'.format(os.path.join(datasets_path, str(d_id), str(d_id)), n_fold))
    test = path_to_dataframe('{0}-10-{1}tst.arff'.format(os.path.join(datasets_path, str(d_id), str(d_id)), n_fold))

    df_types = pd.DataFrame(dict(name=train.columns, type=['categorical' if str(x) == 'category' else 'numerical' for x in train.dtypes]))
    df_types.loc[df_types['name'] == df_types.iloc[-1]['name'], 'type'] = 'target'
    # df = pd.read_csv('../datasets/{0}.csv'.format(d_id))
    # df_types = pd.read_csv('../datasets/{0}_types.csv'.format(d_id))

    for column in train.columns:
        if str(train[column].dtype) == 'category':
            categories = train[column].dtype.categories
            dict_conv = dict(zip(categories, range(len(categories))))
            train.loc[:, column] = train.loc[:, column].replace(dict_conv).astype(np.int32)
            
    for column in test.columns:
        if str(test[column].dtype) == 'category':
            categories = test[column].dtype.categories
            dict_conv = dict(zip(categories, range(len(categories))))
            test.loc[:, column] = test.loc[:, column].replace(dict_conv).astype(np.int32)

    # df_valid = train[~train['target'].isnull()]

    # x_cols = [c for c in df_valid.columns if c != 'target']
    X_train = train[train.columns[:-1]]
    y_train = train[train.columns[-1]]
    X_test = test[test.columns[:-1]]
    y_test = test[test.columns[-1]]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    jvm.stop()

    if queue is not None:
        queue.put((X_train, X_test, y_train, y_test, df_types))

    return X_train, X_test, y_train, y_test, df_types


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
        # with open('log_exp.txt', 'a+') as file_out:
        #     file_out.write("Experience error in pipeline:\n" + str(estimator) + "\n")
        #     file_out.write(str(e) + "\n")
        return None


def fit_predict(estimator, X, y, X_test):
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
        return estimator.predict(X_test)
    except Exception as e:
        with open('log_exp.txt', 'a+') as file_out:
            file_out.write("Experience error in pipeline:\n" + str(estimator) + "\n")
            file_out.write(str(e) + "\n")
        return None


def execute_exp(
        id_trial, n_fold, datasets_path, d_id, n_generations, time_per_task, pool_size,
        mutation_rate_pool, crossover_rate_pool, n_ensembles, mutation_rate_ensemble, crossover_rate_ensemble,
        n_jobs, results_path, seed=None, subsample=1, METRIC='balanced_accuracy', F_METRIC=balanced_accuracy_score):

    # try:
    # with open('log_exp.txt', 'a+') as file_out:
    #     file_out.write(str(d_id)+"_"+str(id_trial)+": " + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + "\n")

    p = AUTOCVEClassifier(
        generations=n_generations,
        population_size_components=pool_size,
        mutation_rate_components=mutation_rate_pool,
        crossover_rate_components=crossover_rate_pool,
        population_size_ensemble=n_ensembles,
        mutation_rate_ensemble=mutation_rate_ensemble,
        crossover_rate_ensemble=crossover_rate_ensemble,
        grammar='grammarTPOT',
        max_pipeline_time_secs=60,
        max_evolution_time_secs=time_per_task,
        n_jobs=n_jobs,
        random_state=seed,
        # scoring=METRIC,  # TODO reactivate
        scoring='balanced_accuracy',
        verbose=1
    )

    mp.set_start_method('spawn')
    ctx = mp.get_context('spawn')

    queue = ctx.Queue()
    pro = mp.Process(
        target=parse_open_ml, kwargs=dict(
            datasets_path=datasets_path, d_id=d_id, n_fold=n_fold, queue=queue
        )
    )
    pro.start()
    pro.join()

    X_train, X_test, y_train, y_test, df_types = queue.get()

    # def handler(signum, frame):
    #     print("Maximum time reached.")
    #     raise SystemExit('Time limit exceeded, sending system exit...')
    #
    # signal.signal(signal.SIGALRM, handler)
    # signal.alarm(time_per_task + GRACE_PERIOD)
    # start = time.time()

    # try:
    p.optimize(X_train, y_train, subsample_data=subsample)
    # except (KeyboardInterrupt, SystemExit) as e:
        # print(e)  # TODO commented
        # raise e

    # duration = time.time()-start

    print('Finished evolutionary process! Hooray!')

    # signal.alarm(0)

    # with open('log_exp.txt', 'a+') as file_out:
    #     file_out.write("Optimization time: "+str(duration)+"\n")

    # move("evolution.log", str(d_id)+"_"+str(id_trial)+"_evolution.log")
    # move("matrix_sim.log", str(d_id)+"_"+str(id_trial)+"_matrix_sim.log")
    # try:
    #     move("evolution_ensemble.log", str(d_id)+"_"+str(id_trial)+"_evolution_ensemble.log")
    # except:
    #     pass

    # with open('pipe_found.txt', 'a+') as file_out:
    #     file_out.write("Problem: "+str(d_id)+", Trial: "+str(id_trial)+"\n\n")

    # try:
    # TODO testing!
    train_probs = fit_predict_proba(p.get_best_pipeline(), X_train, y_train, X_train).astype(np.float64)
    test_probs = fit_predict_proba(p.get_best_pipeline(), X_train, y_train, X_test).astype(np.float64)

    pro = mp.Process(
        target=get_evaluation, kwargs=dict(
            dataset_path=os.path.join(datasets_path, d_id), n_fold=n_fold,
            seed=seed, train_probs=train_probs, test_probs=test_probs,
            results_path=results_path, id_exp=d_id, id_trial=id_trial
        )
    )
    pro.start()
    pro.join()


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
        '--time-per-task', action='store', required=False, type=int, default=7000,
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

    some_args = parser.parse_args()

    if 0 < some_args.n_samples <= 20:
        N_TRIALS = some_args.n_samples
    else:
        raise Exception("The number of trials is expected to be an integer with value between 1 and 20.")

    datasets_names = some_args.datasets_names.split(',')

    results_path = create_metadata_path(args=some_args)
    os.chdir(results_path)

    for id_exp in datasets_names:
        os.mkdir(os.path.join(results_path, id_exp))

        print("Dataset " + id_exp)
        for id_trial in range(1, N_TRIALS + 1):
            print("Trial " + str(id_trial))
            for n_fold in range(1, 10 + 1):  # iterates over folds
                execute_exp(
                    n_fold=n_fold,
                    id_trial=id_trial,
                    datasets_path=some_args.datasets_path,
                    d_id=id_exp,
                    seed=np.random.randint(np.iinfo(np.int32).max),
                    METRIC=unweighted_area_under_roc,
                    n_generations=some_args.n_generations,
                    n_jobs=some_args.n_jobs,
                    time_per_task=some_args.time_per_task,
                    pool_size=some_args.pool_size,
                    mutation_rate_pool=some_args.mutation_rate_pool,
                    crossover_rate_pool=some_args.crossover_rate_pool,
                    n_ensembles=some_args.n_ensembles,
                    mutation_rate_ensemble=some_args.mutation_rate_ensemble,
                    crossover_rate_ensemble=some_args.crossover_rate_ensemble,
                    results_path=results_path
                )

                break  # TODO remove!
            break  # TODO remove!

        print('collapsing!')

        summary = collapse_metrics(os.path.join(results_path, id_exp), only_baselines=True)
        print('summary for dataset %s:' % id_exp)
        print(summary)

    # move('results.txt', 'results_finished.txt')


if __name__ == '__main__':
    main()
