from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


def conf_matrix(test_tensor, pred_tensor, class_list, norm_true=False, norm_predict=False):

    if norm_true:
        cm = np.round(confusion_matrix(test_tensor, pred_tensor, normalize="true"), 2)
    elif norm_predict:
        cm = np.round(confusion_matrix(test_tensor, pred_tensor, normalize="pred"), 2)
    else:
        cm = confusion_matrix(test_tensor, pred_tensor)
    #z = np.flip(cm, 0)
    x = class_list
    y = class_list
    fig = ff.create_annotated_heatmap(z=cm, x=x, y=y, colorscale='Viridis')

    fig.update_layout(title_text='Confusion matrix')
    fig.layout.xaxis.title = "Predicted Label"
    fig.layout.yaxis.title = "True Label"
    fig['layout']['yaxis']['autorange'] = "reversed"

    fig['data'][0]['showscale'] = True

    return fig


def metrics_bar(test_tensor, pred_tensor, class_list):
    cm = confusion_matrix(test_tensor, pred_tensor)
    accur = [(cm[i][i] / sum(cm[i])) for i in range(cm.shape[0])]

    precision = []
    recall = []
    for i in range(cm.shape[0]):
        predict = sum([cm[j][i] for j in range(cm.shape[0])])
        precision.append(cm[i][i]/predict)
        recall.append(cm[i][i]/sum(cm[i]))
    precision_ar = np.array(precision)
    recall_ar = np.array(recall)
    f1 = (2 * precision_ar * recall_ar) / (precision_ar + recall_ar)

    fig = go.Figure()
    fig.add_scatter(x=class_list, y=f1, name="F1 Score")
    fig.add_scatter(x=class_list, y=precision_ar, name="Precision")
    fig.add_scatter(x=class_list, y=recall_ar, name="Recall")
    fig.layout.xaxis.title = "Class"
    fig.update_yaxes(range=[0, 1.2])

    return fig


#
# def release_roc_auc(test_tensor, pred_score_array, model, test_set_path, transform, class_idx):
#
#     def plot_roc_specific(class_idx, classes, fpr, tpr, roc_auc, model):
#         fig = plt.figure()
#         lw = 2
#         plt.plot(fpr[class_idx], tpr[class_idx], color="darkorange", lw=lw,
#                  label="ROC curve (area = %0.2f)" % roc_auc[class_idx])
#         plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel("False Positive Rate")
#         plt.ylabel("True Positive Rate")
#         title_name = "Receiver Operating Characteristic for " + classes[class_idx]
#         plt.title(title_name)
#         plt.legend(loc="lower right")
#
#         plt.close(fig)
#
#     def plot_roc_micro_macro(classes, fpr, tpr, roc_auc, model):
#         # compute micro-average ROC curve and ROC area
#         fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), pred_score_array.ravel())
#         roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#         # compute macro-average ROC curve and ROC area
#         all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
#         mean_tpr = np.zeros_like(all_fpr)
#         for i in range(len(classes)):
#             mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#         # average it and compute AUC
#         mean_tpr /= len(classes)
#
#         fpr["macro"] = all_fpr
#         tpr["macro"] = mean_tpr
#         roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
#         fig = plt.figure()
#         lw=2
#         plt.plot(
#             fpr["micro"],
#             tpr["micro"],
#             label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
#             color="deeppink",
#             linestyle=":",
#             linewidth=4,
#         )
#         plt.plot(
#             fpr["macro"],
#             tpr["macro"],
#             label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
#             color="navy",
#             linestyle=":",
#             linewidth=4,
#         )
#         plt.plot([0, 1], [0, 1], "k--", lw=lw)
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.xlabel("False Positive Rate")
#         plt.ylabel("True Positive Rate")
#         plt.title("Some extension of Receiver operating characteristic to multiclass")
#         plt.legend(loc="lower right")
#
#         plt.close(fig)
#
#     testset = torchvision.datasets.ImageFolder(root=test_set_path, transform=transform)
#     classes = testset.classes
#     classes_labels = [value for value in testset.class_to_idx.values()]
#     test_labels = label_binarize(test_tensor, classes=classes_labels)
#
#     # compute ROC curve and ROC area for each class
#     fpr, tpr, roc_auc = {}, {}, {}
#     for i in range(len(classes)):
#         fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], pred_score_array[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     # plot ROC for specific class and save it
#     plot_roc_specific(class_idx, classes, fpr, tpr, roc_auc, model)
#     # plot average ROC and save it
#     plot_roc_micro_macro(classes, fpr, tpr, roc_auc, model)


# def roc_auc_all(test_tensor, pred_score_array, class_list, class_idx, average=False):


def roc_auc(test_tensor, pred_score_array, class_list, class_idx, average=False, all=False):

    pred_score_array_ar = np.asarray(pred_score_array)
    classes_labels = [i for i in range(len(class_list))]
    test_labels = label_binarize(test_tensor, classes=classes_labels)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(class_list)):
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], pred_score_array_ar[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    if average:
        # compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), pred_score_array_ar.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_list))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(class_list)):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # average it and compute AUC
        mean_tpr /= len(class_list)

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        name_micro = f"micro-average ROC (AUC={roc_auc['micro']:.3f})"
        fig.add_trace(go.Scatter(x=fpr["micro"], y=tpr["micro"], name=name_micro,
                                 line = dict(color='deeppink', width=3, dash='dot')))
        name_macro = f"macro-average ROC (AUC={roc_auc['macro']:.3f})"
        fig.add_trace(go.Scatter(x=fpr["macro"], y=tpr["macro"], name=name_macro,
                                 line=dict(color='darkslateblue', width=3, dash='dash')))

        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain')
        )

        fig.update_layout(legend=dict(x=1, y=0))


        return fig


    elif all:
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        for i in range(len(class_list)):
            name = f"{class_list[i]} (AUC={roc_auc[i]:.3f})"
            fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], name=name, mode='lines'))

        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain')
        )


        fig.update_layout(legend_x=0.95, legend_y=0)

        return fig



    else:
        fig = px.area(
            x=fpr[class_idx], y=tpr[class_idx],
            title=f'ROC for {class_list[class_idx]} (AUC={roc_auc[class_idx]:.3f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate')
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')

        return fig


def falsebar(falseimgs_all_model, result_info_models_dict, class_list):
    alldict = dict(
        models=[],
        true_class=[],
        predicted_class=[]
    )
    models_list = [i for i in result_info_models_dict]
    for i in range(len(falseimgs_all_model)):
        for j in range(len(falseimgs_all_model[i])):
            alldict["models"].append(models_list[i])
            alldict["true_class"].append(class_list[falseimgs_all_model[i][j]["label"]])
            alldict["predicted_class"].append(class_list[falseimgs_all_model[i][j]["prediction"]])
    df = pd.DataFrame(alldict)
    fig = px.histogram(df, x="models", facet_col="true_class", color="predicted_class")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], textangle=25))
    return fig

