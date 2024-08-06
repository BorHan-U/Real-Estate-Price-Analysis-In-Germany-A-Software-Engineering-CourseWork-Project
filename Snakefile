include: "workflow/rules/preprocess.smk"
include: "workflow/rules/analyze.smk"
include: "workflow/rules/evaluate.smk"

# Rule to ensure that all targets are included
rule all:
    input:
        "results/plot_preprocessing/analysis_complete.txt",
        "results/evaluation_model/metrics.csv"

# Rule to handle preprocessing results
rule preprocess_target:
    input:
        "data/preprocessed_data.csv"

# Rule to handle analysis results
rule analyze_target:
    input:
        "results/plot_preprocessing/analysis_complete.txt"

# Rule to handle evaluation results
rule evaluate_target:
    input:
        "results/evaluation_model/metrics.csv"

# Cleanup rule to remove results directories
# Cleanup rule to remove results directories using Python for cross-platform compatibility
rule cleanup:
    shell:
        """
        python -c "import shutil; shutil.rmtree('results/plot_preprocessing'); shutil.rmtree('results/evaluation_model')"
        """
