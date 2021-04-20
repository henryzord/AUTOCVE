import json
import os
from datetime import datetime as dt

import javabridge
import numpy as np
import pandas as pd
import psutil as psutil
from scipy.io import arff
from weka.core import jvm


def create_metadata_path(args):
    should_stop = False
    some_exception = None

    try:
        jvm.start()
        now = dt.now()

        str_time = now.strftime('%Y-%m-%d-%H-%M-%S')

        joined = os.getcwd() if not os.path.isabs(args.metadata_path) else ''
        to_process = [args.metadata_path, str_time]

        # creates metadata path, if nonexistent, and experiment path
        for path in to_process:
            joined = os.path.join(joined, path)
            if not os.path.exists(joined):
                os.mkdir(joined)

        for dataset_name in args.datasets_names.split(','):
            os.mkdir(os.path.join(joined, dataset_name))
            os.mkdir(os.path.join(joined, dataset_name, 'overall'))

        with open(os.path.join(joined, 'parameters.json'), 'w') as f:
            json.dump({k: getattr(args, k) for k in args.__dict__}, f, indent=2)

    except Exception as e:
        some_exception = e
        should_stop = True
    finally:
        jvm.stop()
        if should_stop:
            raise some_exception
        return joined


def path_to_dataframe(dataset_path):
    """
    Reads dataframes from an .arff file, casts categorical attributes to categorical type of pandas.

    :param dataset_path:
    :return:
    """

    value, metadata = path_to_arff(dataset_path)

    df = pd.DataFrame(value, columns=metadata.names())

    for attr_name, attr_type in zip(metadata.names(), metadata.types()):
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
