#include "AUTOCVE.h"
#include "grammar.h"
#include "solution.h"
#include "python_interface.h"
#include "population.h"
#include "population_ensemble.h"
#include "utility.h"
#include <stdlib.h>  
#include <fstream>
#include <sstream>  // stringstream
#include <string.h>
#include <time.h>

#define BUFFER_SIZE 10000


AUTOCVEClass::AUTOCVEClass(
    int seed, int n_jobs, PyObject* timeout_pip_sec, int timeout_evolution_process_sec, char *grammar_file,
    int generations, int size_pop_components, double elite_portion_components, double mut_rate_components,
    double cross_rate_components, int size_pop_ensemble, double elite_portion_ensemble, double mut_rate_ensemble,
    double cross_rate_ensemble,  PyObject *scoring, int cv_folds, int verbose
) {

    this->seed=seed;
    this->n_jobs=n_jobs;
    this->timeout_pip_sec=timeout_pip_sec;
    this->timeout_evolution_process_sec=timeout_evolution_process_sec;
    this->grammar_file=AUTOCVEClass::grammar_file_handler(grammar_file);
    this->generations=generations;
    this->scoring=scoring;
    this->verbose=verbose;
    this->size_pop_components=size_pop_components;
    this->elite_portion_components=elite_portion_components;
    this->mut_rate_components=mut_rate_components;
    this->cross_rate_components=cross_rate_components;
    this->size_pop_ensemble=size_pop_ensemble;
    this->elite_portion_ensemble=elite_portion_ensemble;
    this->mut_rate_ensemble=mut_rate_ensemble;
    this->cross_rate_ensemble=cross_rate_ensemble;

    this->cv_folds=cv_folds;
    this->population=NULL;
    this->grammar=NULL;
    this->interface=NULL;
    this->population_ensemble=NULL;

    this->interface = new PythonInterface(this->n_jobs, this->timeout_pip_sec, this->scoring, this->cv_folds, this->verbose);
}

AUTOCVEClass::~AUTOCVEClass(){
    if(this->population)
        delete this->population;

    if(this->population_ensemble)
        delete this->population_ensemble;

    if(this->grammar)
        delete this->grammar;

    if(this->interface)
        delete this->interface;

    free(this->grammar_file);
   
    Py_XDECREF(this->timeout_pip_sec);
    Py_XDECREF(this->scoring);
}

// main method
int AUTOCVEClass::run_genetic_programming(PyObject *data_X, PyObject *data_y, double subsample_data, int n_classes) {
    time_t start, a1, a2;
    time(&a1);
    time(&start);

    srand(this->seed);

    if(!this->interface->load_dataset(data_X, data_y, subsample_data)) {
        return NULL;
    }

    if(this->grammar) {
        delete this->grammar;
    }
    if(this->population) {
        delete this->population;
    }
    if(this->population_ensemble) {
        delete this->population_ensemble;
    }

    this->grammar = new Grammar(this->grammar_file, this->interface);

    std::ofstream evolution_log;
    evolution_log.open("loggerData.csv");

    if(!evolution_log.is_open()) {
        throw "Cannot create evolution.log file\n";
    }

//    std::ofstream matrix_sim_log;
//    matrix_sim_log.open("matrix_sim.log");
//
//    if(!matrix_sim_log.is_open()) {
//        throw "Cannot create matrix_sim.log file\n";
//    }

//    std::ofstream evolution_ensemble_log;
//    evolution_ensemble_log.open("evolution_ensemble.log");
//
//    if(!evolution_ensemble_log.is_open()) {
//        throw "Cannot create evolution_ensemble.log file\n";
//    }

//    struct timeval start, end;
//    gettimeofday(&start, NULL);

    // population is the population of base classifiers (i.e. trees, Genetic Programming population)
    this->population = new Population(
        this->interface, this->size_pop_components, this->elite_portion_components,
        this->mut_rate_components, this->cross_rate_components, n_classes
    );
    // population_ensemble is the population of ensembles (i.e. binary arrays, Genetic Algorithm population)
    this->population_ensemble = new PopulationEnsemble(
        this->size_pop_ensemble, this->size_pop_components, this->elite_portion_ensemble,
        this->mut_rate_ensemble, this->cross_rate_ensemble, n_classes
        );

    this->population_ensemble->init_population_random();

    int return_flag = this->population->init_population(this->grammar, this->population_ensemble);
    if(!return_flag) {
        return NULL;
    }
    if(return_flag==-1) {
        throw "Population not initialized\n";
    }

    std::string header = "gen,lap time (seconds),";
    std::string pop_str = "nevals (clfs),min (clfs),median (clfs),max (clfs),discarded (clfs),";
    std::string ens_size_str = "min size (ens),median size (ens),max size (ens),";
    std::string ens_str = "nevals (ens),min (ens),median (ens),max (ens),discarded (ens)\n";

    PySys_WriteStdout(header.c_str());
    PySys_WriteStdout(ens_str.c_str());
    evolution_log << header << pop_str << ens_size_str << ens_str;

    std::stringstream firstGenOutput;

    char buffer[32];  // how long is the string to be printed at the terminal

    time(&a2);
    int time_difference = (int)difftime(a2, a1);
    time(&a1);

    evolution_log << 0 << "," << time_difference << ",";
    sprintf(buffer, "%03d              %#5d ", 0, time_difference);
    firstGenOutput << buffer;

    this->population->write_population(0, &evolution_log);
    firstGenOutput << this->population_ensemble->write_population(0, &evolution_log);
    PySys_WriteStdout(firstGenOutput.str().c_str());

    int control_flag;

    int generation_time = 0;
    for(int i = 0; i < this->generations; i++) {

        std::stringstream thisGenOutput;

        if(!(control_flag = this->population->next_generation_selection_similarity(this->population_ensemble))) {
            return NULL;
        }
        this->population_ensemble->next_generation_similarity(this->population);

        time(&a2);
        generation_time = (int)difftime(a2, a1);
        time(&a1);

        evolution_log << i + 1 << "," << generation_time << ",";

        sprintf(buffer, "%03d              %#5d ", i + 1, generation_time);
        thisGenOutput << buffer;

        this->population->write_population(i+1,&evolution_log);
        thisGenOutput << this->population_ensemble->write_population(i+1,&evolution_log);

        PySys_WriteStdout(thisGenOutput.str().c_str());

        // checks timeout
        if(
            this->timeout_evolution_process_sec && (difftime(a2, start)) >=
            this->timeout_evolution_process_sec-generation_time) {
            control_flag=-1;
        }

        //KeyboardException or timeout verified
        if(control_flag == -1) {
            break;
        }
    }

//    PySys_WriteStdout("END PROCESS (%d secs)\n",(int)difftime(end, start));

    evolution_log.close();

    if(!this->interface->unload_dataset()) {
        return NULL;
    }
    return 1;
}

PyObject *AUTOCVEClass::get_best_pipeline(){
    if(!this->population)
        throw "Error: Need to call optimize first.";

    return this->population->get_solution_pipeline_rank_i(0);
}

PyObject *AUTOCVEClass::get_voting_ensemble_all(){
    if(!this->population)
        throw "Error: Need to call optimize first.";

    return this->population->get_population_ensemble_all();
}

PyObject *AUTOCVEClass::get_voting_ensemble_elite(){
    if(!this->population)
        throw "Error: Need to call optimize first.";

    return this->population->get_population_ensemble_elite();
}

PyObject *AUTOCVEClass::get_voting_ensemble_best_mask(){
    if(!this->population || !this->population_ensemble)
        throw "Error: Need to call optimize first.";

    return this->population->get_population_ensemble_mask_i(this->population_ensemble,0);
}


char *AUTOCVEClass::get_grammar_char(){
    if(!this->grammar)
        throw "Error: Need to call optimize first.";

    return this->grammar->print_grammar();
}

char *AUTOCVEClass::get_parameters_char(){
    char *parameters=NULL, buffer[BUFFER_SIZE];

    sprintf(buffer, "%d", this->seed);
    parameters=char_concat(char_concat(parameters, "random_state: "), buffer);

    sprintf(buffer, "%d", this->n_jobs);
    parameters=char_concat(char_concat(parameters, ", n_jobs: "), buffer);

    if(this->timeout_pip_sec==Py_None)
        sprintf(buffer, "%s", "None");
    else
        sprintf(buffer, "%ld", PyLong_AsLong(this->timeout_pip_sec));
    parameters=char_concat(char_concat(parameters, ", max_pipeline_time_secs: "),  buffer);

    sprintf(buffer, "%d", this->timeout_evolution_process_sec);
    parameters=char_concat(char_concat(parameters, ", max_evolution_time_sec: "), buffer);

    parameters=char_concat(char_concat(parameters, ", grammar: "), this->grammar_file);

    sprintf(buffer, "%d", this->generations);
    parameters=char_concat(char_concat(parameters, ", generations: "), buffer);

    sprintf(buffer, "%d", this->size_pop_components);
    parameters=char_concat(char_concat(parameters, ", population_size_components: "), buffer);

    sprintf(buffer, "%.2f", this->mut_rate_components);
    parameters=char_concat(char_concat(parameters, ", mutation_rate_components: "), buffer);

    sprintf(buffer, "%.2f", this->cross_rate_components);
    parameters=char_concat(char_concat(parameters, ", crossover_rate_components: "), buffer);

    sprintf(buffer, "%d", this->size_pop_ensemble);
    parameters=char_concat(char_concat(parameters, ", population_size_ensemble: "), buffer);

    sprintf(buffer, "%.2f", this->mut_rate_ensemble);
    parameters=char_concat(char_concat(parameters, ", mutation_rate_ensemble: "), buffer);

    sprintf(buffer, "%.2f", this->cross_rate_ensemble);
    parameters=char_concat(char_concat(parameters, ", crossover_rate_ensemble: "), buffer);

    PyObject *repr_function, *repr_return;
    if(!(repr_function=PyObject_GetAttrString(this->scoring,(char *)"__repr__")))
        return NULL;
    if(!(repr_return=PyObject_CallObject(repr_function, NULL)))
        return NULL;
    const char *repr_char;
    if(!(repr_char=PyUnicode_AsUTF8(repr_return)))
        return NULL;
    parameters=char_concat(char_concat(parameters, ", scoring: "), repr_char);

    sprintf(buffer, "%d", this->cv_folds);
    parameters=char_concat(char_concat(parameters, ", cv_folds: "), buffer);

    sprintf(buffer, "%d", this->verbose);
    parameters=char_concat(char_concat(parameters, ", verbose: "), buffer);

    return parameters;
}

char* AUTOCVEClass::grammar_file_handler(char *grammar_file_param){
    char *grammar_file_return;

    char sep;

    #ifdef _WIN32
    sep = '\\';  // on windows
    #else
    sep = '/';  // on linux
    #endif

    // if only the name of the grammar is passed, then it will search in the default directory of grammars
    if(!strchr(grammar_file_param, '/') && !strchr(grammar_file_param, '\\')){

        PyObject *autocve_module = PyImport_ImportModule("AUTOCVE.AUTOCVE");
        PyObject *path = PyObject_GetAttrString(autocve_module, "__file__");
        const char *path_char = PyUnicode_AsUTF8(path);

        grammar_file_return = (char*)malloc(sizeof(char)*(strlen(path_char)+1));

        strcpy(grammar_file_return, path_char);
        // char *last_bar=strrchr(grammar_file_return, '/');
        char *last_bar = strrchr(grammar_file_return, sep);
        *last_bar = '\0';  // truncates string

//        grammar_file_return = char_concat(grammar_file_return, "/grammar/");
        #ifdef _WIN32
        grammar_file_return = char_concat(grammar_file_return, "\\grammar\\");
        #else
        grammar_file_return = char_concat(grammar_file_return, "/grammar/");
        #endif
        grammar_file_return=char_concat(grammar_file_return, grammar_file_param);

        Py_XDECREF(path);
        Py_XDECREF(autocve_module); 

    } else {
        grammar_file_return = (char*)malloc(sizeof(char)*(strlen(grammar_file_param)+1));
        strcpy(grammar_file_return, grammar_file_param);
    }

    return grammar_file_return;
}

