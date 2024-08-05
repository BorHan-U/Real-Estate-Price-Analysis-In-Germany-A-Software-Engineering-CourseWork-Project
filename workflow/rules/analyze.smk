rule analyze:
    input:
        "data/preprocessed_data.csv"
    output:
        "results/plot_preprocessing/analysis_complete.txt"
    params:
        output_dir="results/plot_preprocessing"
    shell:
        """
        python workflow/scripts/analyze_data.py {input} {params.output_dir}
        """
