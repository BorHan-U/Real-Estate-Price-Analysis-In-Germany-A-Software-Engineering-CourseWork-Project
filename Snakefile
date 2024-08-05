include: "workflow/rules/preprocess.smk"
include: "workflow/rules/analyze.smk"
include: "workflow/rules/evaluate.smk"

rule all:
    input:
        "results/plot_preprocessing/analysis_complete.txt",
        "results/evaluation_model/metrics.csv"

rule preprocess_target:
    input:
        "data/preprocessed_data.csv"

rule analyze_target:
    input:
        "results/plot_preprocessing/analysis_complete.txt"

rule evaluate_target:
    input:
        "results/evaluation_model/metrics.csv"
