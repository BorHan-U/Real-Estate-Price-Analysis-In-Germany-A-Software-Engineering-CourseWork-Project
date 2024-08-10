# Functional and non-function requirements of the project "Real Estate Price Analysis"

## Requirement Definition

This document details the functional and non-functional requirements for the project, aiming to provide a comprehensive specification that ensures the workflow meets essential criteria for data management, analytical processing, data transformation, result generation, and automation.

## Functional requirements:

1. Data Collection

- Collect data from Kaggle
- import the datasets using Pandas.

2. Preprocessing:

- Handle missing data, outliers, and categorical variables.
- Split the dataset into training and testing sets.

3. Exploratory Data Analysis (EDA):

- Provide statistical summaries of different variables.
- Generate visualizations such as boxplot, heatmap, and correlation matrices.

4. Model Evaluation:

- Evaluate the models using appropriate metrics such as Mean Square Error and R-Squared Score.
- Compare the performance of the three models

UML diagram:

![Activity diagram of the EDA](activity_diagram.svg)

## Non-Functional Requirements:

1. Usability:

- User-friendly, with clear instructions and outputs.

2. Performance:

- Efficiently handle large datasets.
- Provide fast response times during model training and prediction.

3. Reliability:

- Produce consistent and repeatable results across different runs.

4. Maintainability:

- Well-structured code and easy to understand.
- Easy to modify for future enhancements.

5. Documentation:

- All functions and modules should be well-documented.
- Provide a comprehensive user manual or guide.
