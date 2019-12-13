# first, activate the environment in the CONSOLE that will run this script
source activate eda3

# install package with debugging information

# installing package with debug information:
# pip install --no-binary :all: --global-option build --global-option --debug PACKAGE  # TODO example
cd ~/Projects/AUTOCVE
pip install --no-binary :all: --global-option build --global-option --debug .

# debugging

cd ~/anaconda2/envs/eda3/lib/python3.7/site-packages/AUTOCVE
gdb --args python /media/henry/Storage/Henry/Projects/AUTOCVE/AUTOCVE[EACOMP]_script.py --metadata-path /media/henry/Storage/Henry/Projects/trash/AUTOCVE --datasets-path /media/henry/Storage/Henry/Projects/eacomp/keel_datasets_10fcv --datasets-names balancescale --n-generations 4 --n-samples 1 --n-jobs 1 --pool-size 10 --n-ensembles 10

# OR

# target exec python
# run

# attach CLION to GDB process
# then:

# code:
# from AUTOCVE import AUTOCVEClassifier
# import numpy as np
# AUTOCVEClassifier.get_unweighted_area_under_roc(y_true=np.array([0, 1, 0], dtype=int), y_score=np.array([[0.6, 0.4], [0.9, 0.1], [1.0, 0.0]], dtype=float))