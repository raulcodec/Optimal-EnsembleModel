# Optimal-EnsembleModel
Uses predictive output of two ML classifiers and results an optimal ensemble model giving higher or equivalent prediction

Background -
Convex Hull computation gives us Maximum realizable ROC (MRROC) curve on the underlying base classifiers.While convex hull only provides an insight to the extent underlying classifiers can be optimized with it's MRROC , we use this approach to build an optimal classifier using the base classifiers probabilities.The hypothesis is that this ensemble classifier will yield improved or equivalent results to the underlying classifiers at various False Positive Rates (FPR).

Approach -
Inputs to the Ensemble model -
Predicted probabilities from the underlying two classifiers C1 & C2 on a given test set (Tx)
Test data set (Tx) having {t1,t2,t3,...tn} observations on which prediction is carried out by the classifiers
A defined measure on which performance is gauged , here we chose a predefined False Positive Rate → FPRx

Outputs of the Ensemble model -
Prediction of the given input test data set Tx .A set of Prediction → Px {p1,p2,p3...pn}

Metrics around Performance of Ensemble classifier  

Model computation  -
Predicted probabilities from the classifiers C1 & C2 were determined .Here the underlying classifiers were Random Forest Classifier & Light GBM. A predefined measure of FPRx was chosen and corresponding True Positive Rates (TPRs) were determined for the two classifiers at the FPRx. Also for the corresponding Thresholds for each of the Classifier at the given FPRx , binary classification was done .Corresponding performance metrics i.e Confusion matrix were determined.

