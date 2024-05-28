# Predicting Absenteeism at a Danish Super Hospital 🏥 ♥️ 🤖 
### A exam paper in the course Data Science, Predicting, Forecasting at Cognitive Science, Aarhus University by Klara Fomsgaard and Laura Paaby

## Setup
> **Step 1** Run ```setup.sh```

To replicate the results, we have included a bash script that automatically 

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the correct versions of the packages required
4. Runs the script
5. Deactivates the virtual environment

Run the code below in your bash terminal:

```bash
bash setup.sh
```


## Usage
> **Step 2** Run ```prepare_data.py```
The data should be prepared in order to actually predict anythiing :)






## Repository overview
```
.
├── .gitignore
├── .Rprofile                                   <--- script related to environment setup
├── DEC_MAK_EXAM.Rproj
├── jags_output/                                <--- folder containing BUGS objects and data frames containing MPD values for plotting
├── plots/                                      <--- folder containing all plots produced by the scripts in src/
├── README.md
├── renv/                                       <--- folder for storing project environment packages after using renv::restore()
├── renv.lock                                   <--- list of packages automatically added to environment by renv::restore()
└── src/
    ├── Simulations.Rmd                         <--- messy markdown for experimenting with distributions
    ├── group_diff_estimation.R
    ├── group_diff_model_no_reparam.txt         <--- unused Bayesian model
    ├── group_diff_model.txt
    ├── group_diff_recovery.R
    ├── group_mean_estimation.R
    ├── group_model.txt
    ├── group_recovery.R
    ├── plot_functions.R                         <--- collection of all plotting functions utilized across the scripts
    ├── simulation_functions.R                   <--- collection of all data simulation functions utilized across the scripts (mainly for recovery)
    ├── subject_model_norm.txt                   <--- unused Bayesian model
    ├── subject_model.txt
    └── subject_recovery.R
```
