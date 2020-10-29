import numpy
from distutils.core import setup, Extension
import os

AUTOCVE_modulename = Extension('AUTOCVE.AUTOCVE',
                               sources=['src/main.cpp', "src/AUTOCVE.cpp", "src/grammar.cpp", "src/solution.cpp",
                                        "src/population.cpp", "src/python_interface.cpp", "src/population_ensemble.cpp",
                                        "src/utility.cpp"],
                               include_dirs=[numpy.get_include()],
                               language='c++',
                               extra_compile_args=['/std:c++latest' if os.name == 'nt' else '--std=c++latest']
                               # extra_compile_args=['-O0']  # TODO remove when deploying
                               )

setup(
    name="AUTOCVE",
    version="1.0",
    package_dir={'AUTOCVE.util': 'util', 'AUTOCVE.pbil': 'pbil', 'AUTOCVE': '.'},
    packages=["AUTOCVE", "AUTOCVE.util", "AUTOCVE.pbil", "AUTOCVE.util.custom_methods", "AUTOCVE.util.custom_methods.TPOT"],
    description="AUTOCVE is a library that aims to find  good voting ensemble configurations consuming little time.",
    ext_modules=[AUTOCVE_modulename],
    install_requires=['numpy', 'pandas', 'scikit-learn>=0.20.2', 'joblib>=0.13.2', 'PyHamcrest>=1.9.0', 'xgboost>=0.80'],
    author="Celio H. N. Larcher Junior",
    author_email="celiolarcher@gmail.com",
    keywords="AUTO-ML, Machine Learning, Coevolution",
    license="BSD 3-Clause License",
    url="https://github.com/celiolarcher/AUTOCVE/",
    package_data={'AUTOCVE': ['grammar/*']},
    include_package_data=True,
)
