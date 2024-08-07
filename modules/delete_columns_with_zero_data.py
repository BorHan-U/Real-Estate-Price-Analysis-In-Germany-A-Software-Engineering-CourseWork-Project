def delete_columns_with_zero_data(data, threshold):
    for column in data.columns:
        zero_count = (data[column] == 0).sum()
        if zero_count > threshold:
            data = data.drop(column, axis=1)
    return data
