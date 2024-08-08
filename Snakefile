# Include rules from other files
include: "workflow/rules/preprocess.smk"
include: "workflow/rules/analyze.smk"
include: "workflow/rules/evaluate.smk"


# Rule to ensure that all targets are included
rule all:
    input:
        "results/plot_preprocessing/analysis_complete.txt",
        "results/evaluation_model/metrics.csv"
# Rule to create necessary directories before processing
rule create_directories:
    output:
        directory("results/plot_preprocessing"),
        directory("results/evaluation_model")
    run:
        import os
        os.makedirs("results/plot_preprocessing", exist_ok=True)
        os.makedirs("results/evaluation_model", exist_ok=True)

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
        import shutil
        
        def remove_files(directory):
            """Remove all files and directories in the specified directory."""
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    shutil.rmtree(os.path.join(root, name))
        
        def remove_file(file_path):
            """Remove the specified file if it exists."""
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            else:
                print(f"File not found: {file_path}")
        
        # Remove files from the specified directories
        remove_files('results/plot_preprocessing')
        remove_files('results/evaluation_model')
        
        # Remove the specific file from the data folder
        remove_file('data/preprocessed_data.csv')
