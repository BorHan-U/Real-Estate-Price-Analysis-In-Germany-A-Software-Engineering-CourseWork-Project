from setuptools import setup, find_packages

setup(
    name='real estate price analysis',
    version='0.1.0',
    author=['Jayed Akbar Sumon ', 'Mohammed Hasan','Md Borhan Uddin', 'Aijaz Afzaal Ahmed','Md Raju Ahmed'],
    author_email=['sumon@uni-potsdam.de', 'mohammad.hasan@uni-potsdam.de', 'md.borhan.uddin@uni-potsdam.de', 'aijazafzaal.ahmed@uni-potsdam.de', 'ahmed10@uni-potsdam.de'],
    description='Real estate price analysis using various machine learning techniques',
    long_description="The aim of this project is to explore the train dataset by performing preprocessing operations, including data mining techniques like removing irrelevant data and updating missing values. Subsequently, we will apply various machine learning algorithms to the preprocessed dataset. The ultimate goal is to determine the best model for predicting property prices based on their features, including factors like location.",
    url='https://gitup.uni-potsdam.de/sumon/real-estate-price-analysis',
    packages=find_packages(),
    install_requires=[
        'numpy',          
        'pandas',         
        'scikit-learn',
        'lightgbm',
        'seaborn',
        'matplotlib',
        'xgboost'
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='house-prices real-estate-price-prediction machine-learning prediction',
)
