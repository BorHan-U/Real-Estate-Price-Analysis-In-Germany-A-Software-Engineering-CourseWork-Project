def drop_columns_with_zero_threshold(data, threshold):
    zero_counts = (data == 0).sum()
    columns_to_drop = zero_counts[zero_counts > threshold].index
    data = data.drop(columns=columns_to_drop)
    print(zero_counts)
    return data
