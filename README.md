## Musical Genre Classification
During the course of Audio Pattern Recognition, where are covered two main topics how:
- ### Data Mining
- ### Machine Learning
In a specific way related to contexts where are in involved signals.
In this repository there are some python classes created specifically to carry out the project of this academic  year.
The addressed theme is the recognition of musical genres through two learning approaches, supervised and unsupervised.
Various classifiers were used and related to each other in order to evaluate their performance.

Content of the repository:

|-- code/<br/>
|-- Models_Classifier # Main class that given a series of features csv files, allows extract model, test data e plot of metrics.<br/>
|---------- code/utils<br/>
|----------------- featuresToCsv # Used for extract audio features and create a CSV file.<br/>
|----------------- features_functions # Custom functions for extract entropy and entropy energy from audio.<br/>
|----------------- preProcessData # Used for cut all sample in a specific length.<br/>