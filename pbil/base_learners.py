# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# classifiers.py
# Copyright (C) 2014-2019 Fracpete (pythonwekawrapper at gmail dot com)

import traceback

import weka.core.jvm as jvm
from weka.core.classes import from_commandline
from weka.core.converters import Loader
from weka.classifiers import Classifier
import numpy as np


def main():
    # load a dataset
    iris_file = "/home/henry/Projects/eacomp/keel_datasets_10fcv/iris/iris-10-1tra.arff"
    print("Loading dataset: " + iris_file)
    loader = Loader("weka.core.converters.ArffLoader")
    iris_data = loader.load_file(iris_file)
    iris_data.class_is_last()

    # TODO use this line!
    # cmdline = 'weka.classifiers.functions.SMO -K "weka.classifiers.functions.supportVector.NormalizedPolyKernel -E 3.0"'

    classifiers = [
        'weka.classifiers.trees.J48',
        'weka.classifiers.trees.SimpleCart',
        'weka.classifiers.rules.PART',
        'weka.classifiers.rules.JRip',
        'weka.classifiers.rules.DecisionTable',
    ]

    for clf in classifiers:
        # classifier from commandline
        print(clf)

        classifier = from_commandline(clf, classname="weka.classifiers.Classifier")  # type: Classifier
        classifier.build_classifier(iris_data)
        y_scores = classifier.distributions_for_instances(iris_data).astype(np.float64)
        y_pred = y_scores.argmax(axis=1).astype(np.int64)

        print('\tall ok!')

        # print(y_scores)
        # print(y_pred)

    # print(classifier)


if __name__ == '__main__':
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()

