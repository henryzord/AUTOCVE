# AUTOCVE 


This library is intended to be used in the search for hard voting ensembles. Based on a co-evolutionary framework, it 
turns possible to testing multiple ensembles configurations without repetitive training and test procedure of its 
components.

The ensembles created are based on the 
[Voting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) class. 
In the default version, several methods implemented in the [scikit-learn](https://github.com/scikit-learn/scikit-learn) 
package can be used on the final ensemble as well as  the XGBClassifier of the [XGBoost](https://github.com/dmlc/xgboost) 
library. In addition, other [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) 
compatible libraries can be added in a custom grammar. 

### Currently only classification tasks are available (regression tasks coming soon)!

**GECCO experiment scripts can be found in the 
[AUTOCVE_GECCO19](https://github.com/celiolarcher/Experiments_GECCO19) repository.**

## Prerequisites

In this current version, for proper use, it is recommended to install the library in the Anaconda package, since almost 
all dependencies are met in a fresh install. 

This library has not yet been tested on a Windows installation, so correct functionality is not guaranteed. 

For EDNEL, it is required to have a Weka installation.

## Installing

**NOTE TO MYSELF:** If you're having problems compiling the library with `pip` (e.g. could not find Visual Studio 14.0, 
even though it is installed), check if Avast is not moving `vcvarsall.bat` to quarantine.

### For development

This repository uses Anaconda as the default Python. You can download Anaconda [here](https://www.anaconda.com/products/individual).

Follow these steps to set up the development environment for the algorithm:

1. Create a new conda environment: `conda create --name autocve python=3.7.7`
2. Activate it: `conda activate autocve` 
3. Install conda packages: `conda install --file installation/conda_libraries.txt -c conda-forge` 
4. Install JRE and JDK. The correct JDK version is jdk-8u221-linux-x64.tar.gz. Tutorial available 
[here](https://www.javahelps.com/2017/09/install-oracle-jdk-9-on-linux.html).
5. Install pip libraries: `pip install -r installation/pip_libraries.txt` (NOTE: this might require installing Visual 
Studio with Python tools on Windows)
6. Replace Weka from `python-weka-wrapper` library with provided Weka (in installation directory). This is needed since 
SimpleCart is not provided with default Weka. On Weka, simply installing it as an additional package makes it available 
in the GUI; however the wrapper still won't see it.

  * On Ubuntu: 
    
    ```cp installation/weka.jar <folder_to_anaconda_installation>/anaconda3/envs/autocve/lib/python3.7/site-packages/weka/lib/```
    
  * On Windows: 
    
    ```copy installation\weka.jar <folder_to_anaconda_installation>\Anaconda3\envs\autocve\Lib\site-packages\weka\lib\```

7. Install [mPBIL](https://github.com/henryzord/pbil/tree/comparative):

```
git clone --single-branch --branch comparative https://github.com/henryzord/PBIL
cd PBIL
conda activate autocve
python setup.py install
```

8. Install AUTOCVE package: 
```bash
cd AUTOCVE
pip install .
```

## Usage

### As for experiments with EDNEL

```bash
python AUTOCVE[EACOMP]_script.py --metadata-path <metadata_path>
--datasets-path <datasets_path> 
--datasets-names vowel,vehicle,german,diabetic,phoneme,movement_libras,hcvegypt,seismicbumps,drugconsumption,artificialcharacters,twonorm,turkiye,waveform,magic,spambase 
--n-generations 100 --n-samples 10 --n-jobs 1 --pool-size 200 --n-ensembles 200 --heap-size '5G'
```

### As originally intended by the authors

```
from AUTOCVE.AUTOCVE import AUTOCVEClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits=load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

autocve=AUTOCVEClassifier(generations=100, grammar='grammarTPOT', n_jobs=-1)

autocve.optimize(X_train, y_train, subsample_data=1.0)

print("Best ensemble")
best_voting_ensemble=autocve.get_best_voting_ensemble()
print(best_voting_ensemble.estimators)
print("Ensemble size: "+str(len(best_voting_ensemble.estimators)))

best_voting_ensemble.fit(X_train, y_train)
print("Train Score: {:.2f}".format(best_voting_ensemble.score(X_train, y_train)))
print("Test Score: {:.2f}".format(best_voting_ensemble.score(X_test, y_test)))
```

## Procedures

|                  Function |                                                                                               Description |
| ------------------------: | :-------------------------------------------------------------------------------------------------------- | 
|                  optimize | Optimize an ensemble to the (X,y) base. X and y expect to be numeric (used pandas.get_dummies otherwise). |
|  get_best_voting_ensemble |                        Get the best ensemble produced in the optimization procedure (recommended option). |
|         get_best_pipeline |                                                Get the pipeline with higher score in the last generation. |
| get_voting_ensemble_elite |          Get the ensemble compound by the 10% pipelines with higher score defined in the last generation. |
|   get_voting_ensemble_all |                            Get the ensemble compound by all the pipelines defined in the last generation. |
|               get_grammar |                                               Get as text the grammar used in the optimization procedure. |
|            get_parameters |                                            Get as text the parameters used in the optimization procedure. |


## Parameters

All these keyword parameters can be set in the initialization of the AUTOCVE.

| Keyword       | Description|
| ------------- |-------------| 
| random_state                  | seed used in the optimization process | 
| n_jobs                  | number of jobs scheduled in parallel in the evaluation of components   | 
| max_pipeline_time_secs        | maximum time allowed to a single training and test procedure of the cross-validation (None means not time bounded)  |
| max_evolution_time_sec        | maximum time allowed to the whole evolutionary procedure to run (0 means not time bounded)| 
| grammar  | the grammar option or path to a custom grammar used in the Context Free Genetic Program algorithm (used to specfy the algorithms) | 
| generations  | number of generations performed      | 
| population_size_components  | size of the population of components used in the ensembles | 
| mutation_rate_components  | mutation rate of the population of components | 
| crossover_rate_components  | crossover rate of the population of components | 
| population_size_ensemble  | size of the population of ensembles | 
| mutation_rate_ensemble  | mutation rate of the population of ensembles | 
| crossover_rate_ensemble  | crossover rate of the population of ensembles | 
| scoring  | score option used to evaluate the pipelines (sklearn compatible) | 
| cv_folds  | number of folds in the cross validation procedure  | 
| verbose  | verbose option | 


## Contributions

Any suggestions are welcome to improve this work and should be directed to Celio Larcher Junior (celiolarcher@gmail.com).

Despite this, as this work is part of my PhD thesis, the pull request acceptance is limited to simple fixes. 

Also, although I try to continually improve this code, I can not guaranteed an immediate fix of any requested issue.
