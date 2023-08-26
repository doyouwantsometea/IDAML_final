# IDAML_final

 The project aims at applying machine learning methods to predicting forest fire area with the dataset introduced by [Cortez and Morais (2007)](https://repositorium.sdum.uminho.pt/handle/1822/8039). The work was submitted as the final project of the model "Intelligent Data Analysis & Machine Learning" in summer semester 2023 at Universität Potsdam.

 The model adopts ridge regression with kernel trick to make predicts. To improve the performance, data was preprocessed with Z-normalisation, and logistic regression was introduced for filtering out the instances in which the forest was left unburnt.