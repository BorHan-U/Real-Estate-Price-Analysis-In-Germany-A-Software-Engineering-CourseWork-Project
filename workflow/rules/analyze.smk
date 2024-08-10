rule analyze:
    input:
        "data/preprocessed_data.csv"
    output:
        "results/plot_preprocessing/analysis_complete.txt"
    params:
        output_dir="results/plot_preprocessing",
        selected_column="OverallQual"  # Example column, adjust as needed
    shell:
        """
        python workflow/scripts/analyze_data.py {input} {params.output_dir} {params.selected_column}
        """
