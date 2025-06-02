Add general information about the project here... 
Specific information about each model architecture is to be found under each model folder. 


## file structure
```
src/
|-- supervised/
|   |-- base/
|   |   |-- ensemble.py
|   |   |-- knn.py
|   |   |-- linear.py
|   |   |-- svm.py
|   |   |-- tree.py
|   |-- tree/
|   |   |-- decision_tree_classifier.py
|   |   |-- decision_tree_regressor.py
|   |-- ensemble/
|   |   |-- random_forest_classifier.py
|   |   |-- random_forest_regressor.py
|   |   |-- adaboost_classifier.py
|   |   |-- adaboost_regressor.py
|   |   |-- gradient_boost_classifier.py
|   |   |-- gradient_boost_regressor.py
|   |-- linear/
|   |   |-- linear_regression.py
|   |   |-- logistic_regression.py
|   |-- svm/
|   |   |-- svm_classifier.py
|   |   |-- svm_regressor.py
|   |-- knn/
|       |-- knn_classifier.py
|       |-- knn_regressor.py
|
|-- unsupervised/
|   |-- base/
|   |   |-- clustering.py
|   |   |-- decomposition.py
|   |   |-- anomaly.py
|   |   |-- dimensionality.py
|   |-- clustering/
|   |   |-- kmeans.py
|   |   |-- dbscan.py
|   |   |-- gmm.py
|   |-- dimensionality/
|   |   |-- pca.py
|   |   |-- ica.py
|   |   |-- nmf.py
|   |   |-- tsne.py
|   |   |-- isomap.py
|   |   |-- svd.py
|   |-- anomaly/
|       |-- isolation_forest.py
|       |-- one_class_svm.py
|       |-- lof.py
|       |-- robust_covariance.py
```


Other:
agglomerative (clustering)


```
src/
|-- supervised/
|   |-- base/
|   |   |-- tree.py
|   |   |-- linear.py
|   |   |-- ensemble.py
|   |   |-- svm.py
|   |   |-- knn.py
|   |-- classification/
|   |   |-- decision_tree.py
|   |   |-- random_forest.py
|   |   |-- adabost.py
|   |   |-- gradient_boost.py
|   |   |-- knn.py
|   |   |-- logistic_regression.py
|   |   |-- svm.py
|   |-- regression/
|       |-- decision_tree.py
|       |-- random_forest.py
|       |-- adaboost.py
|       |-- gradient_boost.py
|       |-- linear_regression.py
|       |-- svm.py
|
|-- unsupervised/
|   |-- base/
|   |   |-- clustering.py
|   |   |-- decomposition.py
|   |   |-- anomaly.py
|   |   |-- dimensionality.py
|   |-- clustering/
|   |   |-- kmeans.py
|   |   |-- dbscan.py
|   |   |-- gmm.py
|   |-- dimensionality_reduction/
|   |   |-- pca.py
|   |   |-- ica.py
|   |   |-- nmf.py
|   |   |-- tsne.py
|   |   |-- isomap.py
|   |   |-- svd.py
|   |-- anomaly_detection/
|       |-- isolation_forest.py
|       |-- one_class_svm.py
|       |-- lof.py
|       |-- robust_covariance.py
```
