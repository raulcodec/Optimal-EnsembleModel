from scipy.spatial import ConvexHull
import scipy.stats as st
import numpy as np
from future_builtins import zip
import random as rnd
from scipy import *
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib
matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt

def define_confidence_interval(arr_values ,confidence_perct):
    interval_start ,interval_end =st.t.interval(confidence_perct, len(arr_values) - 1, loc=np.mean(arr_values),
                                                 scale=st.sem(arr_values))
    return interval_start, interval_end


def binary_preds(vertex_point,y_pred_clf1,y_pred_clf2,clf_arr):
    if (vertex_point[3] == 1.0):
        thres = vertex_point[2]
        for i in range(len(y_pred_clf1)):
            if (y_pred_clf1[i] > thres):
                clf_arr.append(1)
            else:
                clf_arr.append(0)
    else :
        thres = vertex_point[2]
        for i in range(len(y_pred_clf2)):
            if (y_pred_clf2[i] > thres):
                clf_arr.append(1)
            else:
                clf_arr.append(0)
    return clf_arr


# Requirements : Needs predicted probabilities , Binary classification for the two classifiers
# Output array Hull array : TP ,FP ,Thresholds ,Underlying classifer model for the convex hull vertices
# Output array Metrics Hull points : For Ensemble model computation

def compute_hull_metrics(y_test, y_clf_1, y_clf_2, y_clf1_bin, y_clf2_bin, output_to_file=False):
    fpr_clf1, tpr_clf1, thresholds_clf1 = roc_curve(y_test, y_clf_1)
    auc_clf1 = roc_auc_score(y_test, y_clf_1)

    fpr_clf2, tpr_clf2, thresholds_clf2 = roc_curve(y_test, y_clf_2)
    auc_clf2 = roc_auc_score(y_test, y_clf_2)

    twoD_clf1 = np.vstack((fpr_clf1, tpr_clf1)).T
    twoD_clf2 = np.vstack((fpr_clf2, tpr_clf2)).T

    # Build Convex Hull & stack up the vertices in 2 D array
    all_points = np.concatenate((twoD_clf1, twoD_clf2))
    hull = ConvexHull(all_points)
    # All Hull Vertices
    twoD_ver = np.vstack((all_points[hull.vertices, 0], all_points[hull.vertices, 1])).T

    # Build arrays for model classifier flags for identification
    clf1_cnst = []
    clf2_cnst = []
    for x in range(len(fpr_clf1)):
        clf1_cnst.append(1)
    for x in range(len(fpr_clf2)):
        clf2_cnst.append(2)

    # Stack up metrics -> FPR ,TPR ,Threshold & Model Flags

    metrics_clf1 = np.vstack((fpr_clf1, tpr_clf1, thresholds_clf1, clf1_cnst)).T
    metrics_clf2 = np.vstack((fpr_clf2, tpr_clf2, thresholds_clf2, clf2_cnst)).T

    metrics = np.concatenate((metrics_clf1, metrics_clf2))

    # Compute confusion metrics of the two classifiers

    y_test = y_test.astype("i")
    test_data = np.array(y_test.as_matrix(columns=None))

    cf_clf1 = confusion_matrix(test_data, y_clf1_bin)
    cf_clf2 = confusion_matrix(test_data, y_clf2_bin)

    # Compute TPs & FPs corresponding to the Hull Vertices
    hull_arr = []
    metric_hull_points = []
    for j in range(len(twoD_ver)):
        for i in range(len(metrics)):
            if ((twoD_ver[j][0] == metrics[i][0]) & (twoD_ver[j][1] == metrics[i][1])):
                if (metrics[i][3] == '1.0'):
                    # Metrics to calculate TPx
                    TPR = metrics[i][1]
                    TP = cf_clf1[1][1]
                    FN = cf_clf1[1][0]
                    # Metrics to calculate FPx
                    FPR = metrics[i][0]
                    FP = cf_clf1[0][1]
                    TN = cf_clf1[0][0]

                    TPx = TPR * (TP + FN)
                    FPx = FPR * (FP + TN)

                    hull_metrics = [round(TPx, 2), round(FPx, 2), metrics[i][2], metrics[i][3]]
                    tmp_metr = [metrics[i][0], metrics[i][1], metrics[i][2], metrics[i][3]]

                else:
                    # Metrics to calculate TPx
                    TPR = metrics[i][1]
                    TP = cf_clf2[1][1]
                    FN = cf_clf2[1][0]
                    # Metrics to calculate FPx
                    FPR = metrics[i][0]
                    FP = cf_clf2[0][1]
                    TN = cf_clf2[0][0]

                    TPx = TPR * (TP + FN)
                    FPx = FPR * (FP + TN)

                    hull_metrics = [round(TPx, 2), round(FPx, 2), metrics[i][2], metrics[i][3]]
                    tmp_metr = [metrics[i][0], metrics[i][1], metrics[i][2], metrics[i][3]]

        hull_arr.append(hull_metrics)
        metric_hull_points.append(tmp_metr)

    return hull_arr, metric_hull_points


#Generating Convex Hull plot for underlying two base classifiers
def convex_hull_plot(y_test, y_clf_1, y_clf_2, clf_name1, clf_name2, path, output_to_file=False):
    fpr_clf1, tpr_clf1, thresholds_clf1 = roc_curve(y_test, y_clf_1)
    auc_clf1 = roc_auc_score(y_test, y_clf_1)

    fpr_clf2, tpr_clf2, thresholds_clf2 = roc_curve(y_test, y_clf_2)
    auc_clf2 = roc_auc_score(y_test, y_clf_2)

    twoD_clf1 = np.vstack((fpr_clf1, tpr_clf1)).T
    twoD_clf2 = np.vstack((fpr_clf2, tpr_clf2)).T

    all_points = np.concatenate((twoD_clf1, twoD_clf2))
    hull = ConvexHull(all_points)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_clf1, tpr_clf1, label=clf_name1)
    plt.plot(fpr_clf2, tpr_clf2, label=clf_name2)

    plt.plot(all_points[hull.vertices, 0], all_points[hull.vertices, 1], 'm--', label='Convex hull', lw=2)
    plt.plot(all_points[hull.vertices, 0], all_points[hull.vertices, 1], 'mo')
    plt.plot
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    #plt.title("\nAUROC Score " + str(clf_name1) + ":{0} \n AUROC Score " + str(
    #    clf_name2) + ": {1}\n Area under Hull: {2} \n ".format(auc_clf1, auc_clf2, (hull.volume + 0.5)))
    #plt.legend(loc='best')

    plt.title("AUROC Score model 1: {0}\nAUROC Score model 2: {1} \n Area under Hull: :{2}\n".format(
        auc_clf1, auc_clf2, (hull.volume + 0.5)))
    plt.legend(loc='best')

    plt.savefig(str(path) + "/AUROC_ConvexHull.png", bbox_inches='tight')
    plt.close()
    print str(path)+"/AUROC_ConvexHull.png"
    print "Area under model 1 :"+str(auc_clf1)
    print "Area under model 2 :" + str(auc_clf2 )
    print "Area under Hull: "+str(hull.volume + 0.5)
# Generating Achievable PR plot for the underlying two base classifiers
def achievable_prcurve_plot(y_test, y_pred_clf1, y_pred_clf2, y_pred_clf1_bin, y_pred_clf2_bin,clf_name1, clf_name2, path,
                            output_to_file=False):
    hull_metrics, metric_hull_points = compute_hull_metrics(y_test, y_pred_clf1, y_pred_clf2, y_pred_clf1_bin,
                                                            y_pred_clf2_bin)

    precision_clf1, recall_clf1, thresholds_clf1 = precision_recall_curve(y_test, y_pred_clf1)
    average_precision_clf1 = average_precision_score(y_test, y_pred_clf1, average='micro')

    precision_clf2, recall_clf2, thresholds_clf2 = precision_recall_curve(y_test, y_pred_clf2)
    average_precision_clf2 = average_precision_score(y_test, y_pred_clf2, average='micro')

    # In PR curve , one less value of thrreshold is observed so suffixing a dummy value at end
    thresholds_clf1 = np.append(thresholds_clf1, -999)
    thresholds_clf2 = np.append(thresholds_clf2, -999)

    twoD_clf1 = np.vstack((recall_clf1, precision_clf1)).T
    twoD_clf2 = np.vstack((recall_clf2, precision_clf2)).T

    all_points = np.concatenate((twoD_clf1, twoD_clf2))

    # concatenate_all thresholds of PR

    metrics_clf1 = np.vstack((recall_clf1, precision_clf1, thresholds_clf1)).T
    metrics_clf2 = np.vstack((recall_clf2, precision_clf2, thresholds_clf2)).T
    metrics_all = np.concatenate((metrics_clf1, metrics_clf2))

    # For the array sent from Hull , capture all the thresholds , for the correspoding thresholds in PR curve capture the Prec & Recall
    # These Prec Recall points are the correspoding Max Realizable points from Convex hull
    # Now for these points , capture the corresponding TP & FP  sent in hull array [0] & [1] values and put them in array
    # Sort them in ascending order in
    pr_metrics = 0
    pr_points = []
    pr_recall = []
    pr_prec = []
    dict_TPFP = {}
    for p in range(len(hull_metrics)):
        q = 0
        pr_metrics = 0
        TP = 0.0
        FP = 0.0
        for q in range(len(metrics_all)):
            if (hull_metrics[p][2] == metrics_all[q][2]):
                pr_metrics = [metrics_all[q][0], metrics_all[q][1], metrics_all[q][2]]
                TP = hull_metrics[p][0]
                FP = hull_metrics[p][1]
                # q=q+1
        # print("corresponding pr metrics for"+str(p))
        if (pr_metrics == 0):
            pr_points.append([0.0, 0.0, 0.0])
            pr_recall.append(0.0)
            pr_prec.append(0.0)
            dict_TPFP[TP] = [FP, 0.0, 0.0]
        else:
            pr_points.append(pr_metrics)
            pr_recall.append(pr_metrics[0])
            pr_prec.append(pr_metrics[1])
            # To capture the highest FP for a given TP = 0.0
            dict_TPFP[TP] = [FP, pr_metrics[0], pr_metrics[1]]

    TPFP_points = []
    for key in sorted(dict_TPFP.iterkeys()):
        TPFP_points.append([key, dict_TPFP[key]])
        # print TPFP_points

    # As per the Goadrich paper Each Precx,Recallx point in between the Vertices (Prec,Recall) points is computed as
    # TPa + x/ Tot Pos  ,    TPa + x /(TPa +x +FPa + (FPb-FPa/TPb-TPa)*x)

    # Computing Total positives in the test set
    print "before computing True Positives"
    # Total Positives
    tot_pos = 0
    print "len of ytest : "+str(len(y_test))
    y_test = np.array(y_test.as_matrix(columns=None))
    print y_test
    for i in range(len(y_test)):
        #print y_test[i]
        if (y_test[i] == 1):
            tot_pos = tot_pos + 1
    print "After computing True Positives"
    # Compute the in between Prec & Recall points using the formulae of decaying function
    all_computed_rec = []
    all_computed_prec = []
    # compute all Prec & Recall for TP & FP
    for i in range(len(TPFP_points) - 1):

        value = TPFP_points[i:i + 2]
        TP_1 = value[0][0]
        FP_1 = value[0][1][0]
        rec_1 = value[0][1][1]
        prec_1 = value[0][1][2]

        # print TP_1,FP_1,rec_1,prec_1
        TP_2 = value[1][0]
        FP_2 = value[1][1][0]
        rec_2 = value[1][1][1]
        prec_2 = value[1][1][2]

        # Add the corresponding vetices of Prec & Recall
        all_computed_rec.append(rec_1)
        all_computed_prec.append(prec_1)
        z = TP_2 - TP_1
        z = int(z)
        for i in range(z):
            x = i + 1
            rec_x = (TP_1 + x) / tot_pos
            prec_x = (TP_1 + x) / (TP_1 + x + FP_1 + ((FP_2 - FP_1) / (TP_2 - TP_1)) * x)
            all_computed_rec.append(rec_x)
            all_computed_prec.append(prec_x)

        all_computed_rec.append(rec_2)
        all_computed_prec.append(prec_2)

    # Capturing the max precision for all recall
    dict_pr = {}
    i = 0
    while (i < len(all_points)):
        if all_points[i][0] in dict_pr.keys():
            if (dict_pr[all_points[i][0]] <= all_points[i][1]):
                dict_pr[all_points[i][0]] = all_points[i][1]
        else:
            dict_pr[all_points[i][0]] = all_points[i][1]
        i = i + 1
    # print len(dict_pr)

    # Sorting all the recall points in ascending order with all the highest precision points from the classifiers

    recall_all = []
    prec_all = []
    for key in sorted(dict_pr.iterkeys()):
        recall_all.append(key)
        prec_all.append(dict_pr[key])

    # Plotting the curves
    # print "Area under Achievable curve"
    auc_achievablepr = round(metrics.auc(all_computed_rec, all_computed_prec), 5)

    plt.step(recall_clf1, precision_clf1, color='r', alpha=0.6, where='post', label=clf_name1)
    plt.step(recall_clf2, precision_clf2, color='g', alpha=0.6, where='post', label=clf_name2)

    plt.step(all_computed_rec, all_computed_prec, color='b', alpha=0.6, where='post', label='PR Achievable')
    plt.plot(pr_recall, pr_prec, 'bo')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.title("AUPRC Model 1 Score: {0}\nAUPRC Model 2 Score: {1} \n Achievable PR curve obtained :{2}\n".format(
        average_precision_clf1, average_precision_clf2, auc_achievablepr))
    plt.legend(loc='best')

    plt.savefig(str(path) + "/AchievablePR_Curve.png", bbox_inches='tight')
    plt.close()

# Ensemble Model using base classifiers and their predicted probabilities & binary predicts as input on the FPR chosen
def ensemble_modular(y_test, y_pred_clf1, y_pred_clf2, y_pred_clf1_bin, y_pred_clf2_bin, fpr_chosen,
                     output_to_file=False):
    print " In Ensemble Model"
    hull_arr, metric_hull_points = compute_hull_metrics(y_test, y_pred_clf1, y_pred_clf2, y_pred_clf1_bin,
                                                        y_pred_clf2_bin)

    ###Ensemble model computation
    metric_hull_points = sorted(metric_hull_points, key=lambda x: x[0])

    # np.searchsorted(metric_hull_points[0],fpr_chosen)

    for i in range(len(metric_hull_points)):
        if (fpr_chosen < round(metric_hull_points[i][0], 3)):
            point_before = [metric_hull_points[i - 1][0], metric_hull_points[i - 1][1], metric_hull_points[i - 1][2],
                            metric_hull_points[i - 1][3]]
            point_after = [metric_hull_points[i][0], metric_hull_points[i][1], metric_hull_points[i][2],
                           metric_hull_points[i][3]]
            break

    print "point before"
    print point_before
    print "point after"
    print point_after
    # Calculate ratio of the distance
    before_dist = fpr_chosen - point_before[0]
    after_dist = point_after[0] - fpr_chosen

    ratio = round(before_dist / float(after_dist), 2)
    print "Ratio:"
    print ratio

    # computing prediction at the thresholds of point before & point after w.r.t associated classifier
    # Compute predictions at the Vertices between which metric is associated
    before_pred = []
    after_pred = []
    # Before point all predictions
    before_pred = binary_preds(point_before, y_pred_clf1, y_pred_clf2, before_pred)
    # After point predictions
    after_pred = binary_preds(point_after, y_pred_clf1, y_pred_clf2, after_pred)


    bfpred_count = int(round(1 / (1 + ratio) * len(before_pred)))

    print " Length Before pred > before pred count "
    print "Length of before prediction:" + str(len(before_pred))
    print "Before pred count :" + str(bfpred_count)
    print " Going for 5 iterations for CI"
    # Taking the average value of Five random runs for the given FPR
    fpr_tpr_res = []
    fpr_res = []
    tpr_res = []
    num_of_iter = 5
    for j in range(num_of_iter):
        ensemble_pred = after_pred[:]
        print "Before random sample fn iter: "+str(j)
        for i in rnd.sample(xrange(len(before_pred)), bfpred_count):
            ensemble_pred[i] = before_pred[i]
        # This gives the FPR,TPR,Thresholds at the FPR chosen .Return array contains 3 values , at index 1 is the reqd metric
        fpr_ensemble, tpr_ensemble, thresholds_ensemble = roc_curve(y_test, ensemble_pred)
        fpr_res.append(fpr_ensemble[1])
        tpr_res.append(tpr_ensemble[1])
        fpr_tpr_res.append([fpr_ensemble[1], tpr_ensemble[1]])

    # define confidence interval for acceptable TPR & FPR based on the range of values obtained
    fpr_int_strt, fpr_int_end = define_confidence_interval(fpr_res, 0.95)
    tpr_int_strt, tpr_int_end = define_confidence_interval(tpr_res, 0.95)

    print "All 10 iterations done"
    while (True):
        # Get an ensemble model prediction
        result_pred = after_pred[:]
        for i in rnd.sample(xrange(len(before_pred)), bfpred_count):
            result_pred[i] = before_pred[i]
        fpr_result, tpr_result, thresholds_result = roc_curve(y_test, result_pred)

        # Check if the results obtained are within the confidence interval ? If yes then exit , else continue
        if ((fpr_result[1] >= fpr_int_strt) & (fpr_result[1] <= fpr_int_end)):
            if ((tpr_result[1] >= tpr_int_strt) & (tpr_result[1] <= tpr_int_end)):
                break

    # Converting y_test data as numpy array
    y_test = y_test.astype("i")
    test_data = np.array(y_test.as_matrix(columns=None))

    cf_result = confusion_matrix(test_data, result_pred)
    ensemble_measure = np.vstack((fpr_result[1], tpr_result[1], thresholds_result[1])).T


    print "Mean Value of iterations"
    avg_tpr = 0
    avg_fpr = 0
    for i in range(len(fpr_tpr_res)):
        avg_fpr = avg_fpr + fpr_tpr_res[i][0]
        avg_tpr = avg_tpr + fpr_tpr_res[i][1]

    return ensemble_measure,result_pred