import argparse
import inspect
import multiprocessing as mp
import shutil
from datetime import datetime as dt
from functools import reduce
import itertools as it

from weka.classifiers import Classifier

from pbil import generation
from pbil.evaluations import EDAEvaluation, collapse_metrics, __check_missing__
from pbil.individuals import Skeleton, Individual, ClassifierWrapper
from pbil.model import PBIL
from utils import *

# so final results are always reported
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 300)


def process_fold_sample(
        n_sample, n_fold, n_individuals, n_generations, learning_rate, selection_share, datasets_path,
        variables_path, classifiers_path, this_path, seed, n_jobs=1, heap_size='2G', cheat=False, only_baselines=False
    ):

    some_exception = None

    clfs = [x[0] for x in inspect.getmembers(generation, inspect.isclass)]
    classifier_names = [x for x in clfs if ClassifierWrapper in eval('generation.%s' % x).__bases__]

    try:
        if n_jobs > 1:
            jvm.start(max_heap_size=heap_size)

        classifiers = json.load(open(classifiers_path, 'r'))
        variables = json.load(open(variables_path, 'r'))

        train_data, test_data = read_datasets(datasets_path, n_fold)

        subfolder_path = os.path.join(this_path, 'sample_%02.d_fold_%02.d' % (n_sample, n_fold))

        if os.path.exists(subfolder_path):
            shutil.rmtree(subfolder_path)
            os.mkdir(subfolder_path)

        ens_names = ['baseline', 'RandomForest']
        jobjects = []
        if not only_baselines:
            pbil = PBIL(lr=learning_rate, n_generations=n_generations,
                        n_individuals=n_individuals, selection_share=selection_share,
                        classifier_names=classifier_names, classifier_data=classifiers, variables=variables,
                        train_data=train_data, subfolder_path=subfolder_path, test_data=None if not cheat else test_data
                        )

            overall, last = pbil.run(seed)

            ens_names += ['overall', 'last']
            jobjects += [overall._jobject_ensemble, last._jobject_ensemble]

            pbil.logger.individual_to_file(individual=overall, individual_name='overall', step=pbil.n_generation)
            pbil.logger.individual_to_file(individual=last, individual_name='last', step=pbil.n_generation)
            pbil.logger.probabilities_to_file()

            # noinspection PyUnusedLocal
            baseline = Individual.from_baseline(
                seed=seed, classifiers=[clf for clf in classifier_names if overall.log[clf]],
                train_data=train_data, test_data=None if not cheat else test_data
            )
        else:
            baseline = Individual.from_baseline(
                seed=seed, classifiers=['J48', 'SimpleCart', 'PART', 'JRip', 'DecisionTable'],
                train_data=train_data, test_data=None if not cheat else test_data
            )

        random_forest = Classifier(classname='RandomForest')
        random_forest.build_classifier(train_data)

        jobjects += [baseline._jobject_ensemble, random_forest.jobject]

        dict_models = dict()
        for ens_name, jobject in zip(ens_names, jobjects):
            train_evaluation, test_evaluation = Skeleton.from_sets(
                jobject_ensemble=jobject, train_data=train_data, test_data=test_data, seed=seed
            )

            dict_metrics = dict()
            for metric_name, metric_aggregator in EDAEvaluation.metrics:
                value = getattr(test_evaluation, metric_name)
                if isinstance(value, np.ndarray):
                    new_value = np.array2string(value.ravel().astype(np.int32), separator=',')
                    new_value_a = 'np.array(%s, dtype=np.%s).reshape(%s)' % (new_value, value.dtype, value.shape)
                    value = new_value_a

                dict_metrics[metric_name] = value

            df_metrics = pd.DataFrame(dict_metrics, index=[ens_name])
            dict_models[ens_name] = df_metrics

        collapsed = reduce(lambda x, y: x.append(y), dict_models.values())
        collapsed.to_csv(os.path.join(
                this_path, 'overall',
                'test_sample-%02.d_fold-%02.d.csv' % (n_sample, n_fold)), index=True
            )

    except Exception as e:
        some_exception = e
    finally:
        if n_jobs > 1:
            jvm.stop()
        if some_exception is not None:
            raise some_exception


def __get_running_processes__(jobs):
    if len(jobs) > 0:
        jobs[0].join()

        for job in jobs:  # type: mp.Process
            if not job.is_alive():
                jobs.remove(job)

    return jobs


def start_processes(combs, n_jobs, kwargs):
    jobs = []
    for n_sample, n_fold in combs:
        if len(jobs) >= n_jobs:
            jobs = __get_running_processes__(jobs)

        job = mp.Process(
            target=process_fold_sample,
            args=(n_sample, n_fold),
            kwargs=kwargs
        )
        job.start()
        jobs += [job]

    # blocks everything
    for job in jobs:
        job.join()


def main(args):
    assert os.path.isdir(args.datasets_path), ValueError('%s does not point to a datasets folder!' % args.datasets_path)

    datasets_names = args.datasets_names.split(',')

    now = dt.now()

    mp.set_start_method('spawn')
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = mp.Process(
        target=metadata_path_start, kwargs=dict(
            now=now, args=args, datasets_names=datasets_names, queue=queue
        )
    )
    p.start()
    p.join()

    these_paths = queue.get()

    # TODO allow running next datasets before!

    for this_path, dataset_name in zip(these_paths, datasets_names):
        tensorboard_start(this_path=this_path, launch_tensorboard=args.launch_tensorboard)

        combs = it.product(range(1, args.n_samples + 1), range(1, 11))

        some_exception = None

        kwargs = dict(
            seed=args.seed,
            n_generations=args.n_generations,
            n_individuals=args.n_individuals,
            learning_rate=args.learning_rate,
            selection_share=args.selection_share,
            datasets_path=os.path.join(args.datasets_path, dataset_name),
            variables_path=args.variables_path,
            classifiers_path=args.classifiers_path,
            this_path=this_path,
            n_jobs=args.n_jobs,
            heap_size=args.heap_size,
            cheat=args.cheat,
            only_baselines=args.only_baselines
        )

        if args.n_jobs == 1:
            try:
                jvm.start(max_heap_size=args.heap_size)
                for n_sample, n_fold in combs:
                    kwargs['n_sample'] = n_sample
                    kwargs['n_fold'] = n_fold

                    process_fold_sample(**kwargs)
            except Exception as e:
                some_exception = e
            finally:
                jvm.stop()
                if some_exception is not None:
                    raise some_exception

        else:
            start_processes(combs, args.n_jobs, kwargs)

        # repeteco time
        # missing = __check_missing__(path=os.path.join(this_path, 'overall'))
        # start_processes(missing, args.n_jobs, kwargs)

        try:
            if args.n_jobs > 1:
                jvm.start()
        except:
            pass

        try:
            summary = collapse_metrics(os.path.join(this_path, 'overall'), only_baselines=args.only_baselines)
            print('summary for dataset %s:' % dataset_name)
            print(summary)

            jvm.stop()
        except Exception as e:
            jvm.stop()
            if is_debugging():
                raise e
            else:
                print('could not collapse metrics for dataset %s' % dataset_name)
                print('Reason:\n%s' % str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Main script for running Estimation of Distribution Algorithms for ensemble learning.'
    )

    parser.add_argument(
        '--datasets_path', action='store', required=True,
        help='Must lead to a path that contains several subpaths, one for each dataset. Each subpath, in turn, must '
             'have the arff files.'
    )

    parser.add_argument(
        '--datasets-names', action='store', required=True,
        help='Name of the datasets to run experiments. Must be a list separated by a comma '
             'Example:\n'
             'python script.py --datasets-names iris,mushroom,adult'
    )

    parser.add_argument(
        '--classifiers-path', action='store', required=True,
        help='Path to file with classifiers and their hyper-parameters.'
    )

    parser.add_argument(
        '--metadata-path', action='store', required=True,
        help='Path to folder where runs results will be stored.'
    )

    parser.add_argument(
        '--variables-path', action='store', required=True,
        help='Path to folder with variables and their parameters.'
    )

    parser.add_argument(
        '--seed', action='store', required=False, default=np.random.randint(np.iinfo(np.int32).max),
        help='Seed used to initialize base classifiers (i.e. Weka-related). It is not used to bias PBIL.', type=int
    )

    parser.add_argument(
        '--n-jobs', action='store', required=False, default=1,
        help='Number of jobs to use. Will use one job per sample per fold. '
             'If unspecified or set to 1, will run in a single core.',
        type=int
    )
    parser.add_argument(
        '--heap-size', action='store', required=False, default='2G',
        help='string that specifies the maximum size, in bytes, of the memory allocation pool. '
             'This value must be a multiple of 1024 greater than 2MB. Append the letter k or K to indicate kilobytes, '
             'or m or M to indicate megabytes. Defaults to 2G'
    )

    parser.add_argument(
        '--cheat', action='store_true', required=False, default=False,
        help='Whether to log test metadata during evolution.'
    )

    parser.add_argument(
        '--n-generations', action='store', required=True,
        help='Maximum number of generations to run the algorithm', type=int
    )

    parser.add_argument(
        '--n-individuals', action='store', required=True,
        help='Number of individuals in the population', type=int
    )

    parser.add_argument(
        '--n-samples', action='store', required=True,
        help='Number of times to run the algorithm', type=int
    )

    parser.add_argument(
        '--learning-rate', action='store', required=True,
        help='Learning rate of PBIL', type=float
    )

    parser.add_argument(
        '--selection-share', action='store', required=True,
        help='Fraction of fittest population to use to update graphical model', type=float
    )

    parser.add_argument(
        '--launch-tensorboard', action='store', required=False, default=False,
        help='Whether to launch tensorboard.'
    )

    parser.add_argument(
        '--only-baselines', action='store_true', required=False, default=False,
        help='Only run baseline algorithms.'
    )

    some_args = parser.parse_args()

    main(args=some_args)
