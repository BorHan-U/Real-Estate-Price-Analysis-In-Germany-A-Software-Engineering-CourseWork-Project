rule evaluate:
    input:
        "data/preprocessed_data.csv"
    output:
        "results/evaluation_model/metrics.csv"
    params:
        output_dir="results/evaluation_model"
    shell:
        """
        python workflow/scripts/evaluate_models.py {input} {params.output_dir}
        """
