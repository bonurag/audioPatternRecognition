## Musical Genre Classification
During the course of Audio Pattern Recognition, where are covered two main topics how:
- ### Data Mining
- ### Machine Learning
In a specific way related to contexts where are in involved signals.
In this repository there are some python classes created specifically to carry out the project of this academic  year.
The addressed theme is the recognition of musical genres through two learning approaches, supervised and unsupervised.
Various classifiers were used and related to each other in order to evaluate their performance.

### Content of the repository:
1. code:
     - **Models_Classifier:** Main class that allows to extracting model, test data e plot metrics.
     - **Test_Model:** Class used to test the chosen classifier model.
     - **apr_constants:** Class that contains the constants and fix path used in project.
     - **apr_functions_sl:** Class that contains functions used only for supervised learning approach.
     - **apr_functions_ul:** Class that contains functions used only for unsupervised learning approach.
     - **common_functions:** Class that contains functions used only for both approach.
     - **plot_functions:** Class that contains functions for plot graph. 
2. code/utils:
    - **featuresToCsv:** Used for extract audio features and create a CSV file.
    - **features_functions:** Custom functions for extract entropy and entropy energy from audio.
    - **preProcessData:** Used for cut all sample in a specific length.
3. code/notebooks:
    - **Verify_KMeans_Model:** Jupyter used for the unsupervised approach evaluation.
    - **Verify_RF_Model:** Jupyter used for the supervised approach evaluation.     
4. report: 
    - **Project_2021___Musical_Genre_Classification:** Project report

### Plugins:
For making this project were used this following libraries.

| Plugin | README |
| ------ | ------ |
| seaborn | https://seaborn.pydata.org/ |
| pandas | https://pandas.pydata.org/ |
| librosa | https://librosa.org/doc/latest/index.html |
| matplotlib | https://matplotlib.org/ |
| numpy | https://numpy.org/ |
| tqdm | https://pypi.org/project/tqdm/ |
| pydub | https://pypi.org/project/pydub/ |
| scikit-learn | https://scikit-learn.org/stable/ |