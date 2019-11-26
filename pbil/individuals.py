import copy

from weka.core.classes import Random

from pbil.evaluations import EDAEvaluation, internal_5fcv, evaluate_on_test
from pbil.generation import *
from pbil.integration import baseline_aggregator_options
from pbil.registry import PBILLogger


class Skeleton(object):
    def __init__(self, seed, log, options, fitness, test_unweighted_auc=None):
        """

        :param log: Hyper-parameters used for generating this individual
        :type log: dict
        """

        self.seed = seed
        self.log = log
        self.options = options
        self.fitness = fitness
        self.test_unweighted_auc = test_unweighted_auc

    def __deepcopy__(self, memodict={}):
        ind = Skeleton(
            seed=copy.deepcopy(self.seed),
            log=copy.deepcopy(self.log),
            options=copy.deepcopy(self.options),
            fitness=copy.deepcopy(self.fitness),
            test_unweighted_auc=copy.deepcopy(self.test_unweighted_auc)
        )
        return ind

    @staticmethod
    def from_sets(seed, jobject_ensemble, train_data, test_data=None):

        train_evaluation_obj = internal_5fcv(seed=seed, jobject=jobject_ensemble, train_data=train_data)
        train_pevaluation = EDAEvaluation.from_jobject(train_evaluation_obj, data=train_data, seed=seed)

        if test_data is not None:
            test_evaluation_obj = evaluate_on_test(jobject=jobject_ensemble, test_data=test_data)
            test_pevaluation = EDAEvaluation.from_jobject(test_evaluation_obj, data=test_data, seed=seed)
        else:
            test_pevaluation = None

        return train_pevaluation, test_pevaluation

    def __str__(self):
        return str(self.fitness)


class Individual(Skeleton):
    def __init__(self, seed, log, options, train_data, test_data=None):
        """

        :param log:
        :param options:
        :param train_data:
        :param test_data:
        """

        _jobject_ensemble = Individual.__set_jobject_ensemble__(options=options, train_data=train_data)

        train_evaluation, test_evaluation = Skeleton.from_sets(
            jobject_ensemble=_jobject_ensemble, train_data=train_data, test_data=test_data, seed=seed
        )

        super(Individual, self).__init__(
            seed=seed, log=log, options=options,
            fitness=train_evaluation.unweighted_area_under_roc,
            test_unweighted_auc=test_evaluation.unweighted_area_under_roc
        )

        self.train_evaluation = train_evaluation
        self.test_evaluation = test_evaluation
        self._jobject_ensemble = _jobject_ensemble
        self._train_data = train_data
        self._test_data = test_data

        self.classifiers = self.__initialize_classifiers__()

    def __deepcopy__(self, memodict={}):
        ind = Individual(
            seed=copy.deepcopy(self.seed),
            log=copy.deepcopy(self.log),
            options=copy.deepcopy(self.options),
            train_data=self._train_data,
            test_data=self._test_data
        )
        return ind

    def __initialize_classifiers__(self):
        env = javabridge.get_env()  # type: javabridge.JB_Env
        clf_objs_names = env.get_object_array_elements(javabridge.call(self._jobject_ensemble, 'getClassifiersNames', '()[[Ljava/lang/String;'))

        clfs = []
        for t in clf_objs_names:
            obj_name, obj_class, obj_sig = list(map(javabridge.to_string, env.get_object_array_elements(t)))
            if len(self.options[obj_class]) > 0:
                obj = javabridge.get_field(self._jobject_ensemble, obj_name, obj_sig)

                clf = eval(obj_class).from_jobject(obj)
                clfs += [clf]

        return clfs

    @staticmethod
    def __set_jobject_ensemble__(options, train_data):
        opts = []
        for flag, listoptions in options.items():
            opts.extend(['-' + flag, ' '.join(listoptions)])

        eda_ensemble = javabridge.make_instance(
            'Leda/EDAEnsemble;', '([Ljava/lang/String;Lweka/core/Instances;)V', opts, train_data.jobject
        )
        return eda_ensemble

    def predict(self, data):
        raise NotImplementedError('not implemented yet!')

    def predict_proba(self, data):
        """

        :param data:
        :type data: weka.core.datasets.Instances
        :return:
        """
        env = javabridge.get_env()  # type: javabridge.JB_Env

        if len(data) == 1:
            dist = javabridge.call(
                self._jobject_ensemble, 'distributionForInstance', '(Lweka/core/Instance;)[D', data.jobject)
            final_dist = env.get_double_array_elements(dist)
        else:
            dist = javabridge.call(
                self._jobject_ensemble, 'distributionsForInstances', '(Lweka/core/Instances;)[[D', data.jobject
            )
            final_dist = np.array([env.get_double_array_elements(x) for x in env.get_object_array_elements(dist)])

        return final_dist

    def __str__(self):
        raise NotImplementedError('not implemented yet!')
        clf_texts = ''
        for clf in self.classifiers:
            clf_texts += '%s\n\n' % str(clf)
        return clf_texts

    @classmethod
    def from_baseline(cls, seed, classifiers, train_data, test_data=None):
        options, ilog = baseline_classifiers_options(classifiers)
        aggoptions, agglog = baseline_aggregator_options(None)
        options.update(aggoptions)
        ilog.update(agglog)

        ind = Individual(seed=seed, log=ilog, options=options, train_data=train_data, test_data=test_data)
        return ind

    @staticmethod
    def cold_evaluation(probs, y_true):
        preds = probs.argmax(axis=1)

        metrics_dict = dict()
        for metric_name, func in PBILLogger.metrics:
            metrics_dict[metric_name] = func(y_true=y_true, y_pred=preds, probs=probs)

        return metrics_dict

