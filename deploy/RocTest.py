# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc

def plot_roc(y_score, y_test, titlt):
    ###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
    # y_score = [0.1, 0.5, 0.8, 0.6, 0.3]
    # y_test = [0, 1, 1, 0, 0]

    # Compute ROC curve and ROC area for each class
    fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.5, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(titlt)
    plt.legend(loc="lower right")
    plt.show()
    # plt.savefig(save_path)