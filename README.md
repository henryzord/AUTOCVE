# AUTOCVE-star

This is a heavily modified version of AUTOCVE, by Henry Cagnini.

* The original version can be found [here](https://github.com/celiolarcher/AUTOCVE/).
* The experiments performed by the original authors, for their 2019 GECCO paper, can be found [here](https://github.com/celiolarcher/Experiments_GECCO19). 
* The paper that describes the original algorithm, AUTOCVE, can be found [here](https://dl.acm.org/doi/10.1145/3321707.3321844).

This code is intended to be used in a nested cross-validation procedure, which is not the original design of the algorithm by the authors.

---

## Modifications

The table below describes all modifications made to the original algorithm.

|                       | [AUTOCVE](https://github.com/celiolarcher/AUTOCVE/) |                            AUTOCVE-star (this repository) |
|:----------------------|----------------------------------------------------:|----------------------------------------------------------:|
| Intention             | Holdout evaluation with fixed hyper-parameters      | Nested cross-validation with hyper-parameter optimization |
| Fitness function      | Balanced accuracy                                   | Unweighted AUC over all classes (even if binary problem)  |
| Base classifiers      | 11 (refer to paper for complete list)               | 5: J48, SimpleCart, JRIP, PART, Decision Table            |
| Aggregation function  | Simple Majority Voting                              | Simple Majority Voting                                    |
| Data transformations? | Yes, 4 types (refer to paper for complete list)     | No                                                        |

The reader should consider all other aspects of the algorithm (mutation policy, crossover policy, etc) equal.

## Installation

**NOTE:** If you're having problems compiling the library with `pip` (e.g. could not find Visual Studio 14.0, 
even though it is installed), check if Avast is not moving `vcvarsall.bat` to quarantine.

1. Download Python Anaconda from [here](https://www.anaconda.com/products/individual).
2. Create a new conda environment: 
   
    `conda create --name autocve python=3.7.7`
   
3. Activate it:
   
    `conda activate autocve`
   
4. Install conda packages:
   
   `conda install --file installation/conda_libraries.txt -c conda-forge`

5. Install JRE and JDK. The correct JDK version is jdk-8u261-linux-x64.tar.gz. Tutorial available 
[here](https://www.javahelps.com/2017/09/install-oracle-jdk-9-on-linux.html).
6. Install pip libraries (NOTE: this might require installing Visual 
Studio with Python tools on Windows): 
   
    `pip install -r installation/pip_libraries.txt` 
   
7. Replace Weka from `python-weka-wrapper` library with provided Weka (in installation directory). This is needed since 
SimpleCart is not provided with default Weka, and some functionalities are added to the default .jar. Here the .jar is provided,
however the source code is [here](https://github.com/henryzord/WekaCustom/tree/comparative). 
   
  * On Ubuntu: 
    
    ```cp installation/weka.jar <folder_to_anaconda_installation>/anaconda3/envs/autocve/lib/python3.7/site-packages/weka/lib/```
    
  * On Windows: 
    
    ```copy installation\weka.jar <folder_to_anaconda_installation>\Anaconda3\envs\autocve\Lib\site-packages\weka\lib\```

8. Install AUTOCVE-star: 
```bash
cd AUTOCVE
conda activate autocve
pip install .
```

## Usage

* **Note:** Use only one dataset per call.
* The maximum number of usable jobs is 10.

```bash
python nestedcv_autocve.py --heap-size 4g --datasets-path <datasets_path> --dataset-name <dataset_name> 
--metadata-path <metadata_path> --n-internal-folds 5 --n-jobs 10
```