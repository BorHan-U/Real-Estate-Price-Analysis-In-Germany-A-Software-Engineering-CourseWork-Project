# Research Software Engineering Group Project

## Topic: Real Estate Price Analysis

Authors: Aijaz Afzaal Ahmed, Jayed Akbar Sumon, Md Borhan Uddin, Mohammad Hasan, Md Raju Ahmed

### Overview

This research project was worked on during the group project part of the course **Research Software Engineering** at the University of Potsdam.

We aim to develop a machine learning model to predict the real estate (housing) prices while using several input parameters as the base for our prediction.

## Installation

To use this script, you need to have Python installed on your system (3.8 or higher). You also need to install the required dependencies by running the following command:

```
pip install -r requirements.txt
```

## Usage

The script requires two input files: house.py and train.csv. Place these files in the data directory of your project. Then, run the following command to execute the preprocessing script:

```
python bin/house-price.py data/train.csv
```

The script will perform the following steps:

1. Data Preprocessing: The script performs data preprocessing, including handling missing values and transforming numerical features.
2. Data Visualization: It generates histograms and box plots to visualize the data distribution and relationships between variables.
3. Model Evaluation: The script evaluates multiple machine learning models (Linear Regression, Random Forest, and LGBM) and provides Mean Squared Error (MSE) and R-squared scores.
4. Results: The evaluation results are saved in the results/ directory, including yPred*yTrue_table*{model_name}.txt files.

## Testing

Testing is being done either on its own with pytest:

```
pytest .\modules\test_model_evaluation.py .\modules\test.py
```

this just give the test results and if any of them failed or passed.

For combination of coverage with testing we are using:

```
coverage run -m pytest .\modules\test_model_evaluation.py .\modules\test.py
```

from the project directory.

This gives us a .coverage file. It can be used to display the results with either:

```
coverage report -m
```

for a console result or

```
coverage html
```

which will create a htmlcov folder containing an `index.html` file that can be opened and the content viewed in a web browser of your choice.

## Integrating and Using Snakemake
This project uses Snakemake to manage and automate the data preprocessing and model evaluation workflows. Snakemake ensures that the steps in the workflow are executed in the correct order and only re-executed if necessary. Below are instructions on how to set up and run Snakemake for this project.

Before you can use Snakemake, ensure you have the following installed:
```
pip install snakemake
```
To run the workflow, navigate to the directory containing the Snakefile and run Snakemake:
```
snakemake --cores 1
```
## Directory Structure

Make sure your project directory has the following structure:

```
project/
├── bin/
│   ├── house-price.py
│   └── house-price-analysis.ipynb
├── data/
│   ├── train.csv
│   └── data_desciption.txt
├── docs/
│   ├── UML_diagram.png
│   ├── component_analysis.pdf
│   └── requirements.md   
├── modules/
│   ├── modules.py
│   ├── test_model_evaluation.py
│   └── test.py
├── package/
│   ├── build/
│   │   └── bin/
    |   └── modules/
│   ├── dist/
│   │   └── real-estate-price-analysis.0.1.0.tar.gz
│   ├── House_Prices.egg-info/
│   └── setup.py
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
Feel free to modify the project structure and adjust the code according to your specific requirements.

## Requirements

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

You can install these packages by running the command mentioned in the "Installation" section.

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

## Examples

Example usage of functions in the `modules.py` script can be found in the [`house-price.py`](bin/house-price.py) script.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
