import functools
import json
import os
import signal
import subprocess
import sys
import time
import warnings
import webbrowser

import javabridge
import numpy as np
import pandas as pd
import psutil as psutil
from scipy.io import arff
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
from weka.core import jvm
from weka.core.converters import Loader
from weka.core.dataset import Instances


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


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

@deprecated
def binarize(df):

    class_label = df.columns[-1]

    for column in df.columns[:-1]:  # ignores class
        if str(df[column].dtype) == 'category':
            data = df[column].astype(np.object)
            fact, fact_names = pd.factorize(data)
            fact = fact[:, np.newaxis]
            enc = OneHotEncoder(categories='auto').fit(fact)
            transformed = enc.transform(fact).toarray()
            for i, category in enumerate(fact_names):
                # adds a column with the category name to the dataframe
                df['_'.join([column, category])] = transformed[:, i]

            del df[column]  # removes old column from dataframe

    # reorders columns
    _set = list(df.columns)
    _set.remove(class_label)
    _set += [class_label]

    df = df[_set]

    return df

@deprecated
def binarize_all(datasets_path):
    """
    Binarizes all datasets in a given folder.

    :param datasets_path:
    :return:
    """
    if os.path.isdir(datasets_path):
        for dataset in os.listdir(datasets_path):
            if '.arff' in dataset:
                df = path_to_dataframe(os.path.join(datasets_path, dataset))
                df = binarize(df)

                _new_name = (dataset.split('.')[0]) + '_binarized.csv'

                df.to_csv(os.path.join(datasets_path, _new_name), index=False)
                print('done for %s' % dataset.split('.')[0])


def is_weka_compatible(path_dataset):
    """
    Checks whether a dataset is in the correct format for being used by Weka. Throws an exception if not.

    :param path_dataset: Path to dataset. Must be in .arff format.
    """

    ff = arff.loadarff(open(path_dataset, 'r'))  # checks if file opens in scipy

    p = subprocess.Popen(
        ["java", "-classpath", "/home/henry/weka-3-8-3/weka.jar", "weka.classifiers.rules.ZeroR", "-t", path_dataset],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    rc = p.returncode

    if rc != 0:
        raise Exception('Weka could not process correctly the dataset.')


def to_java_object(val, dtype):
    """
    Casts a Python object to a Java object.

    :param val: actual value
    :param dtype: Numpy dtype
    :return: the corresponding Java object
    """

    # signatures are available here:
    # https://pythonhosted.org/javabridge/highlevel.html

    if dtype == 'np.bool':
        javaobj = javabridge.make_instance("java/lang/Boolean", "(Z)V", val)
    elif dtype == 'np.float64':
        javaobj = javabridge.make_instance("java/lang/Float", "(D)V", val)
    elif dtype == 'np.int64':
        javaobj = javabridge.make_instance('java/lang/Integer', "(I)V", val)
    elif dtype == 'np.object':
        javaobj = javabridge.make_instance('java/lang/String', "(Ljava/lang/String;)V", val)
    elif dtype == 'pass':
        javaobj = val
    else:
        raise TypeError('unsupported type:', dtype)

    return javaobj


def from_python_stringlist_to_java_stringlist(matrix):
    env = javabridge.get_env()  # type: javabridge.JB_Env
    # finding array's length

    # creating an empty array of arrays
    jarr = env.make_object_array(len(matrix), env.find_class('[Ljava/lang/String;'))
    # setting each item as an array of int row by row
    for i in range(len(matrix)):
        arrayobj = env.make_object_array(len(matrix[i]), env.find_class('Ljava/lang/String;'))
        for j in range(len(matrix[i])):
            env.set_object_array_element(
                arrayobj, j,
                javabridge.make_instance('Ljava/lang/String;', '(Ljava/lang/String;)V', matrix[i][j])
            )
        env.set_object_array_element(jarr, i, arrayobj)

    return jarr


def read_datasets(dataset_path, n_fold):
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

    return train_data, test_data


def find_process_pid_by_name(process_name):
    """
    Returns PID of process if it is alive and running, otherwise returns None.
    Adapted from https://thispointer.com/python-check-if-a-process-is-running-by-name-and-find-its-process-id-pid/
    """

    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name'])
            # Check if process name contains the given name string.
            if process_name.lower() == pinfo['name'].lower():
                return pinfo['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    return None


def train_test_val_split_index(data, train_size=0.90):
    dist = data.values(data.class_attribute.index)

    warnings.simplefilter(action='ignore', category=FutureWarning)
    train_index, val_index = model_selection.train_test_split(
        np.arange(len(data)), shuffle=True, stratify=dist, train_size=train_size
    )
    return train_index, val_index


def train_test_val_split(data, train_size=0.90):
    """
    Divides a dataset into train and test subsets.

    :param train_size: Size of the training set after the split.
    :param data:
    :type data: weka.core.dataset.Instances
    :rtype: tuple
    :return: train_data, val_data, test_data
    """

    train_index, test_index = train_test_val_split_index(data, train_size=train_size)

    train_data = data.create_instances(name=data.classname, atts=data.attributes(), capacity=len(train_index))
    for i, g in enumerate(train_index):
        inst = data.get_instance(g)
        inst.weight = 1.
        train_data.add_instance(inst, i)
    train_data.class_is_last()

    test_data = data.create_instances(name=data.classname, atts=data.attributes(), capacity=len(test_index))
    for i, g in enumerate(test_index):
        inst = data.get_instance(g)
        inst.weight = 1.
        test_data.add_instance(inst, i)
    test_data.class_is_last()

    # resti_stats = rest_data.attribute_stats(rest_data.class_index)
    # train_stats = train_data.attribute_stats(train_data.class_index)
    # valid_stats = val_data.attribute_stats(val_data.class_index)
    # testi_stats = test_data.attribute_stats(test_data.class_index)
    #
    # sets_stats = pd.DataFrame(data=[
    #     (np.array(resti_stats.nominal_counts) / resti_stats.total_count).tolist() + [resti_stats.total_count],
    #     (np.array(train_stats.nominal_counts) / train_stats.total_count).tolist() + [train_stats.total_count],
    #     (np.array(valid_stats.nominal_counts) / valid_stats.total_count).tolist() + [valid_stats.total_count],
    #     (np.array(testi_stats.nominal_counts) / testi_stats.total_count).tolist() + [testi_stats.total_count]],
    #     index=['rest', 'train', 'validation', 'test'],
    #     columns=train_data.attribute(train_data.class_index).values + ['total']
    # )
    # print(sets_stats)

    return train_data, test_data


def metadata_path_start(now, args, datasets_names, queue=None):
    jvm.start()

    str_time = now.strftime('%d-%m-%Y-%H:%M:%S')

    joined = os.getcwd() if not os.path.isabs(args.metadata_path) else ''
    to_process = [args.metadata_path, str_time]

    for path in to_process:
        joined = os.path.join(joined, path)
        if not os.path.exists(joined):
            os.mkdir(joined)

    with open(os.path.join(joined, 'parameters.json'), 'w') as f:
        json.dump({k: getattr(args, k) for k in args.__dict__}, f, indent=2)

    these_paths = []
    for dataset_name in datasets_names:
        local_joined = os.path.join(joined, dataset_name)
        these_paths += [local_joined]

        if not os.path.exists(local_joined):
            os.mkdir(local_joined)
            os.mkdir(os.path.join(local_joined, 'overall'))

        y_tests = []
        class_name = None
        for n_fold in range(1, 11):
            train_data, test_data = read_datasets(os.path.join(args.datasets_path, dataset_name), n_fold)
            y_tests += [test_data.values(test_data.class_attribute.index)]
            class_name = train_data.class_attribute.name

        # concatenates array of y's
        pd.DataFrame(
            np.concatenate(y_tests),
            columns=[class_name]
        ).to_csv(os.path.join(local_joined, 'overall', 'y_test.txt'), index=False)

    jvm.stop()

    if queue is not None:
        queue.put(these_paths)

    return joined


def tensorboard_start(this_path, launch_tensorboard):
    if (not is_debugging()) and launch_tensorboard:
        pid = find_process_pid_by_name('tensorboard')
        if pid is not None:
            os.kill(pid, signal.SIGKILL)

        p = subprocess.Popen(
            ["tensorboard", "--logdir", this_path, "--port", "default"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        time.sleep(1)
        webbrowser.open_new_tab("http://localhost:6006")


def is_debugging():
    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        return False
    elif gettrace():  # in debug mode
        return True
    return False


def macro_fpr_tpr(y_true, probs):
    """
    The code for generating macro AUC was extracted from
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    :param y_true: ground truth labels.
    :param probs: must be a probability matrix where each row is a instance and each column the probability of
        the class.
    """

    class_indices = list(range(probs.shape[1]))

    fpr = dict()
    tpr = dict()

    if len(class_indices) == 2:  # binary case
        class_indices.remove(0)

    for i in class_indices:
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(np.int32), probs[:, i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in class_indices]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in class_indices:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(class_indices)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr

    return fpr["macro"], tpr["macro"]


def macro_auc(y_true, probs):
    return auc(*macro_fpr_tpr(y_true, probs))


def mean_macro_auc(y_trues, probss):
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(y_trues)):
        y_true = y_trues[i]
        probs = probss[i]

        fpr, tpr = macro_fpr_tpr(y_true=y_true, probs=probs)
        tprs += [np.interp(mean_fpr, fpr, tpr)]
        tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    return mean_auc


