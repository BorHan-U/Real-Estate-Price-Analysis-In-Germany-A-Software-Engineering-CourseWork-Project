import numpy as np

def apply_1_plus_log_transformation(data, columns_to_transform):
    transformed_data = data.copy()
    for column in columns_to_transform:
        transformed_data[column] = np.log1p(transformed_data[column])
    return transformed_data
