import copy

import gc
from deap import tools
from deap.tools import HallOfFame

# noinspection PyUnresolvedReferences
from pbil.generation import *
# noinspection PyUnresolvedReferences
from pbil.integration import *
from pbil.evaluations import EDAEvaluator
from pbil.individuals import Skeleton, Individual
from pbil.ptypes import process_update
from pbil.registry import PBILLogger
from utils import *


class EarlyStop(object):
    def __init__(self):
        self.n_early_stop = 10
        self.tolerance = 0.005
        self.last_bests = np.linspace(0, 1, num=self.n_early_stop)

    def is_stopping(self):
        # return np.all((self.last_bests - self.last_bests.mean()) < self.tolerance)
        return abs(self.last_bests.max() - self.last_bests.min()) < self.tolerance

    def update(self, halloffame, gen):
        self.last_bests[gen % self.n_early_stop] = halloffame[0].fitness


class PBIL(object):
    train_scalars = {
        'fitness_mean': lambda population, last, overall: np.mean([x.fitness for x in population]),
        'fitness_last': lambda population, last, overall: last.fitness,
        'fitness_overall': lambda population, last, overall: overall.fitness
    }

    test_scalars = {
        'test_mean': lambda population, last, overall: np.mean([x.test_unweighted_auc for x in population]),
        'test_last': lambda population, last, overall: last.test_unweighted_auc,
        'test_overall': lambda population, last, overall: overall.test_unweighted_auc
    }

    def __init__(self,
                 lr, n_generations, n_individuals, selection_share, variables, classifier_names, classifier_data,
                 train_data, subfolder_path, test_data=None
                 ):

        self.lr = lr  # type: float
        self.selection_share = selection_share  # type: float
        self.n_generations = n_generations  # type: int
        self.n_individuals = n_individuals  # type: int
        self.classifier_names = classifier_names  # type: list
        self.variables = variables  # type: dict
        self.classifier_data = classifier_data  # type: dict
        self.test_data = test_data  # type: Instances
        self.train_data = train_data  # type: Instances
        self.n_classes = len(self.train_data.class_attribute.values)

        self.evaluator = EDAEvaluator(n_folds=5, train_data=self.train_data, test_data=self.test_data)

        self.n_generation = 0

        scalars = copy.deepcopy(self.train_scalars)
        if self.test_data is not None:
            scalars.update(self.test_scalars)

        self.logger = PBILLogger(logdir_path=subfolder_path, histogram_names=['fitness'],
                                 scalars=scalars, text_names=['last', 'overall']
                                 )

        # register first probabilities
        self.logger.log_probabilities(variables=self.variables)

    def sample_and_evaluate(self, seed, n_individuals, halloffame):
        """

        :param seed:
        :param n_individuals:
        :param halloffame:
        :type halloffame: deap.tools.support.HallOfFame
        :return:
        """

        len_hall = len(halloffame)

        if len_hall == 0:
            parameters = {k: [] for k in self.classifier_names}
            parameters['Aggregator'] = []
            ilogs = []
        else:
            parameters = {k: [x.options[k] for x in halloffame] for k in self.classifier_names}
            parameters['Aggregator'] = [x.options['Aggregator'] for x in halloffame]
            ilogs = [x.log for x in halloffame]
            halloffame.clear()

        for j in range(n_individuals):
            ilog = dict()

            for classifier_name in self.classifier_names:
                ilog[classifier_name] = np.random.choice(
                    a=self.variables[classifier_name]['params']['a'],
                    p=self.variables[classifier_name]['params']['p']
                )
                if ilog[classifier_name]:  # whether to include this classifier in the ensemble
                    options, cclog = eval(classifier_name).sample_options(
                        variables=self.variables, classifiers=self.classifier_data
                    )

                    ilog.update(cclog)
                    parameters[classifier_name] += [options]
                else:
                    parameters[classifier_name].append([])

            ilog['Aggregator'] = np.random.choice(
                a=self.variables['Aggregator']['params']['a'], p=self.variables['Aggregator']['params']['p']
            )
            agg_options, alog = eval(ilog['Aggregator']).sample_options(variables=self.variables)
            ilog.update(alog)

            parameters['Aggregator'] += [[ilog['Aggregator']] + agg_options]

            ilogs += [ilog]

        train_aucs, test_aucs = self.evaluator.get_unweighted_aucs(seed=seed, parameters=parameters)

        # hall of fame is put in the front
        for i in range(0, len_hall):
            local_options = {k: parameters[k][i] for k in self.classifier_names}
            local_options['Aggregator'] = parameters['Aggregator'][i]
            halloffame.insert(Skeleton(
                seed=seed,
                log=ilogs[i],
                options=local_options,
                fitness=train_aucs[i],
                test_unweighted_auc=test_aucs[i]
            ))
        population = []
        for i in range(len_hall, n_individuals + len_hall):
            local_options = {k: parameters[k][i] for k in self.classifier_names}
            local_options['Aggregator'] = parameters['Aggregator'][i]
            population += [Skeleton(
                seed=seed,
                log=ilogs[i],
                options=local_options,
                fitness=train_aucs[i],
                test_unweighted_auc=test_aucs[i]
            )]

        return halloffame, population

    def update(self, population, halloffame):
        self.logger.log_probabilities(variables=self.variables)
        self.logger.log_population(population=population, halloffame=halloffame)

        # selects fittest individuals
        _sorted = sorted(zip(population, [ind.fitness for ind in population]), key=lambda x: x[1], reverse=True)
        population, fitnesses = zip(*_sorted)
        fittest = population[:int(len(population) * self.selection_share)]
        observations = pd.DataFrame([fit.log for fit in fittest])

        # update classifiers probabilities
        for variable_name, variable_data in self.variables.items():
            self.variables[variable_name] = process_update(
                ptype=variable_data['ptype'], variable_name=variable_name, variable_data=variable_data,
                observations=observations, lr=self.lr, n_generations=self.n_generations
            )

        self.n_generation += 1

    def run(self, seed):
        # Statistics computation
        stats = tools.Statistics(lambda ind: ind.fitness)
        for stat_name, stat_func in PBILLogger.population_operators:
            stats.register(stat_name, stat_func)

        hof = HallOfFame(maxsize=1)
        best_last, logbook = self.__run__(
            seed=seed, ngen=self.n_generations, stats=stats, verbose=True, halloffame=hof
        )

        best_overall = hof[0]  # type: Individual

        gc.collect()

        return best_overall, best_last

    def __run__(self, seed, ngen, halloffame, stats=None, verbose=__debug__):
        """This is algorithm implements the ask-tell model proposed in
        [Colette2010]_, where ask is called `generate` and tell is called `update`.

        :param ngen: The number of generation.
        :type ngen: int
        :param stats: A :class:`~deap.tools.Statistics` object that is updated
                      inplace, optional.
        :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                           contain the best individuals, optional.
        :type halloffame: deap.tools.support.HallOfFame
        :param verbose: Whether or not to log the statistics.
        :returns: The final population
        :rtype: A class:`~deap.tools.Logbook` with the statistics of the
                  evolution

        The algorithm generates the individuals using the :func:`toolbox.generate`
        function and updates the generation method with the :func:`toolbox.update`
        function. It returns the optimized population and a
        :class:`~deap.tools.Logbook` with the statistics of the evolution. The
        logbook will contain the generation number, the number of evalutions for
        each generation and the statistics if a :class:`~deap.tools.Statistics` is
        given as argument. The pseudocode goes as follow ::

            for g in range(ngen):
                population = toolbox.generate()
                evaluate(population)
                toolbox.update(population)

        .. [Colette2010] Collette, Y., N. Hansen, G. Pujol, D. Salazar Aponte and
           R. Le Riche (2010). On Object-Oriented Programming of Optimizers -
           Examples in Scilab. In P. Breitkopf and R. F. Coelho, eds.:
           Multidisciplinary Design Optimization in Computational Mechanics,
           Wiley, pp. 527-565;

        """
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        early = EarlyStop()

        population = []
        for gen in range(ngen):
            # early stop
            if early.is_stopping():
                break

            # Generate a new population, already evaluated; re-evaluates halloffame with new seed
            halloffame, population = self.sample_and_evaluate(
                seed=seed, n_individuals=self.n_individuals, halloffame=halloffame
            )

            halloffame.update(population)

            # Update the strategy with the evaluated individuals
            self.update(population=population, halloffame=halloffame)

            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(population), **record)
            if verbose:
                print(logbook.stream)

            early.update(halloffame=halloffame, gen=gen)

        fitnesses = [ind.fitness for ind in population]

        best_skeleton = population[int(np.argmax(fitnesses))]  # type: Skeleton
        best_last = Individual(
            seed=seed, log=best_skeleton.log,
            options=best_skeleton.options, train_data=self.train_data, test_data=self.test_data
        )

        skts = [halloffame[i] for i in range(len(halloffame))]
        halloffame.clear()
        for i in range(len(skts)):
            ind = Individual(
                seed=seed, log=skts[i].log,
                options=skts[i].options, train_data=self.train_data, test_data=self.test_data
            )
            halloffame.insert(ind)

        return best_last, logbook
