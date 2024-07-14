# Snakefile

# Define paths to input data and scripts
DATA_DIR = "data/"
TRAIN_DATA = DATA_DIR + "train.csv"

SCRIPT_DIR = "bin/"
PREPROCESS_SCRIPT = SCRIPT_DIR + "house-price.py"

RESULTS_DIR = "results/"
PREPROCESS_PLOT_DIR = RESULTS_DIR + "plot_preprocessing/"

# Rule for preprocessing the data
rule preprocess:
    input:
        TRAIN_DATA
    output:
        PREPROCESS_PLOT_DIR + "transformed_data_histogram_plot.png"
    params:
        output_dir=PREPROCESS_PLOT_DIR
    shell:
        "python {PREPROCESS_SCRIPT} {input} {params.output_dir}"

# Define the overall workflow
rule all:
    input:
        PREPROCESS_PLOT_DIR + "transformed_data_histogram_plot.png"
