rule preprocess:
    input:
        "data/train.csv"
    output:
        "data/preprocessed_data.csv"
    params:
        output_dir="results/plot_preprocessing"
    shell:
        """
        python workflow/scripts/preprocess_data.py {input} {output} {params.output_dir}
        """
