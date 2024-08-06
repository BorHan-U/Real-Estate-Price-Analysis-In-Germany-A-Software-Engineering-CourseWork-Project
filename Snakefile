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

# Cleanup rule to remove results directories using Python for cross-platform compatibility
rule cleanup:
    run:
        import os
        
        def remove_files(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    os.remove(os.path.join(root, file))
        
        remove_files('results/plot_preprocessing')
        remove_files('results/evaluation_model')
    