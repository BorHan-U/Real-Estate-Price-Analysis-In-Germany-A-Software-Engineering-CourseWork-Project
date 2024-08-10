# Research Software Engineering Group Project 2

The project (Real Estate Price Analysis) aims to analysis of house price data and develop a machine learning model to predict the real estate (housing) prices while using several input parameters as the base for our prediction.

This research project was worked on during the group project part of the course **Research Software Engineering**.

Authors: Aijaz Afzaal Ahmed, Jayed Akbar Sumon, Md Borhan Uddin, Mohammad Hasan, Md Raju Ahmed

## Table of contents
- [Introduction](#introduction)
- [Usage](#usage)
    - [Package](#package-requirement)
    - [Running the Workflow](#running-the-workflow)
    - [Testing](#testing)
    - [Integrating and Using Snakemake](#integrating-and-using-snakemake)
- [Directory Structure](#directory-structure)
- [Functions](#functions)
    - [Examples Usage of Function](#examples-usage-of-function)
- [Contributing Guidelines](#contributing-guidelines)
- [Contact Information](#contact-information)
- [Citation](#citation)
- [License](#license)


## Introduction
Finding a house becomes a daunting task in many countries including germany. To alleviate this problem, we aim to develop a machine learning model to predict the real estate (housing) prices while using several input parameters as the base for our prediction. It seeks to explore and identify the key factors influencing property prices using an extensive dataset of property attributes. In brief, by employing Exploratory data analysis, the result uncovered visualizations, patterns and correlations within the data, ultimately predicting sale prices and highlighting the most influential factors.

## User Stories
As a student researcher, we want to perform exploratory data analysis on the housing dataset and implement feature engineering techniques, so that we can identify key factors influencing house prices and improve the predictive power of our model.

As a project team member, we want to develop and compare multiple machine learning models (e.g., linear regression, random forest, gradient boosting) for house price prediction, so that we can select the most effective algorithm and demonstrate our understanding of advanced regression techniques.

As a data science student, we want to implement cross-validation techniques and analyze our model for potential biases, so that we can ensure the reliability, generalizability, and fairness of our house price predictions.

As a team member responsible for data visualization and presentation, we want to create insightful graphs and charts and prepare a clear summary of our methodology and results, so that we can effectively communicate our findings to professors and peers during the final presentation.

As a project coordinator, we want to set up a version control system for our code and documentation and discuss potential real-world applications of our house price prediction model, so that we can collaborate effectively throughout the project and demonstrate its practical relevance to the real estate industry.

## Usage

Ensure you have the following prerequisites installed on your system:

- [Python3](https://www.python.org/downloads/): For environment management.
- [Snakemake](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html): For workflow management.
- [VSCode](https://code.visualstudio.com/download): For IDE, can be chosen on preference.

## Package Requirement
The script requires the following Python packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- coverage
- lightgbm
- scipy
- pillow
- xgboost
- snakemake

You can install these packages by running the following command.

```sh
pip install -r requirements.txt
```


## Running the Workflow

For the first run, please create the result folder running the following command:

```sh
snakemake --cores all create_directories
```
After creating `results` directory, to run the entire workflow, execute the following command from the root directory of the repository:

```sh
snakemake --cores all
```

To clean up the results (plots, .txt files in results folder), run the following command:

```sh
snakemake --cores all cleanup
```

Note: If the results folder already contains necessary files then the workflow will not rerun, instead it will show that there is nothing to execute as the expected files are already available in the results folder. So, for fresh rerun cleanup first using cleanup script.

## Running Specific Steps
There are three individual steps in the porject.
1. Preprocessing the data
2. Analysis of the data 
3. Model evaluation

These steps can be run individually. As the steps are internally dependedent on each other, before running step 2 or 3, please first run the fir step which is `Preprocess_data`.

To run the `Preprocess_data` step, the command is as follows,

```sh
snakemake --cores all preprocess_target
```
Then  for other two steps, the commands are:
```sh
snakemake --cores all analyze_target
```
and
```sh
snakemake --cores all evaluate_target
```


## Testing
For testing, run the following command from root directory

```sh
pytest
```

this just give the test results and if any of them failed or passed.

For combination of coverage with testing we are using:

```sh
coverage run -m pytest 
```

from the project directory.

This gives us a .coverage file. It can be used to display the results with either:

```sh
coverage report -m
```

for a console result or

```sh
coverage html
```

which will create a htmlcov folder containing an `index.html` file that can be opened and the content viewed in a web browser of your choice.

## Integrating and Using Snakemake
This project uses Snakemake to manage and automate the data preprocessing and model evaluation workflows. Snakemake ensures that the steps in the workflow are executed in the correct order and only re-executed if necessary. Below are instructions on how to set up and run Snakemake for this project.

Before you can use Snakemake, ensure you have the following installed:
```sh
pip install snakemake
```
To run the workflow, navigate to the directory containing the Snakefile and run Snakemake:
```sh
snakemake --cores  all
```

### Command Line Interface (CLI) Usage

The files in the `modules` folder can be executed directly from the command line. Below are the instructions for running each script:

### count_null_data.py
Counts and prints the number of missing values in each column of the dataset.

```sh
python modules/count_null_data.py data/train.csv
```

### delete_columns_with_zero_data.py
Removes columns with a high number of zero values from the dataset.

```sh
python modules/drop_columns_with_zero_threshold.py data/train.csv 400 --output data/filtered_data_for_zero_threshold.csv
```
Use the above scripts structure to and proper function arguments for the other files.

## Directory Structure

Make sure your project directory has the following structure:

```
project/
├── data/
│   ├── train.csv
│   └── data_desciption.txt
├── docs/
│   ├── UML_diagram.png
│   ├── component_analysis.pdf
│   └── requirements.md   
├── modules/
│   └── apply_1_plus_log_transformation.py 
│   ├── count_null_data.py
│   └── delete_columns_with_zero_data.py
│   ├── drop_columns_with_zero_threshold.py
│   └── hyperparameter_tuning.p
│   ├── model_evaluation.py
│   └── plot_boxplot.py 
│   ├── plot_categorical_columns.py
│   └── plot_heatmaps.py
│   ├── separate_categorical_numerical.py
├── package/
│   ├── build/
│   │   └── bin/
│   |   └── modules/
│   ├── dist/
│   │   └── real-estate-price-analysis.0.1.0.tar.gz
│   ├── House_Prices.egg-info/
│   └── setup.py
├── tests/
│   └── test_apply_1_plus_log_transformation.py 
│   ├── test_count_null_data.py
│   └── test_delete_columns_with_zero_data.py
│   ├── test_drop_columns_with_zero_threshold.py
│   └── test_hyperparameter_tuning.p
│   ├── test_model_evaluation.py
│   └── test_plot_boxplot.py 
│   ├── test_plot_categorical_columns.py
│   └── test_plot_heatmaps.py
│   ├── test_separate_categorical_numerical.py
├── workflow/
│   ├── rules
│   │   └── analyze.smk
│   │   └── evaluate.smk
│   │   └── preprocess.smk
│   ├── scripts
│   │   └── analyze_data.py
│   │   └── evaluate_models.py
│   │   └── preprocess_data.py
├── results/
│   ├── plot_preprocessing
│   │   └── [<graph>.png]
│   └── evaluation_model
│       └── [<prediction>.txt]
├── CHANGELOG.md
├── citation.cff
├── CONDUCT.md
├── CONTRIBUTING.md
├── Snakefile
├── LICENSE.txt
├── README.md
├── .gitignore
└── requirements.txt

```


## Functions

The modules.py script contains utility functions used by the house-price.py script to preprocess data and evaluate models:

<ul>
    <li><b>count_null_data</b>: Counts and prints the number of missing values in each column of the dataset.</li>
    <li><b>delete_columns_with_zero_data</b>: Removes columns with a high number of zero values from the dataset.</li>
    <li><b>separate_categorical_numerical</b>: Separates categorical and numerical columns in the dataset.</li>
    <li><b>drop_columns_with_zero_threshold</b>: Drops columns with a high number of zero values based on the specified threshold.</li>
    <li><b>plot_categorical_columns</b>: Plots bar charts for categorical columns to visualize value counts.</li>
    <li><b>apply_1_plus_log_transformation</b>: Applies the 1 plus log transformation to specified numerical columns.</li>
    <li><b>model_evaluation</b>: Evaluates machine learning models with hyperperameter tuning and returns the Mean Squared Error (MSE) and R-squared scores.</li>
</ul>


## Contributing Guidelines
If you wish to contribute to the project, please review the  [contribution guidelines](CONTRIBUTING.md) and the  [code of conduct](CONDUCT.md) . By participating, you are expected to adhere to these guidelines.

## Contact Information

For any inquiries, please contact us at:

- [Jayed Akbar Sumon](mailto:jayed.akbar.sumon@uni-potsdam.de)
- [Aijaz Afzaal Ahmed](mailto:aijaz.ahmed@uni-potsdam.de)
- [Md Borhan Uddin](mailto:md.borhan.uddin@uni-potsdam.de)
- [Mohammad Hasan](mailto:hasan3@uni-potsdam.de)
- [Md Raju Ahmed](mailto:ahmed10@uni-potsdam.de)

## Citation

For information on how to cite this project, please refer to the [Citation file](citation.cff).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
Sure, I can help you with that. To call the scripts in the `modules` folder from the command line, you can create a section in your README.md that explains how to do this. Here is an example of how you can structure this section:

