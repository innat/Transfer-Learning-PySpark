
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.GnBu):

                          plt.imshow(cm, interpolation='nearest', cmap=cmap)
                          plt.title(title)
                          tick_marks = np.arange(len(classes))
                          plt.xticks(tick_marks, classes, rotation=45)
                          plt.yticks(tick_marks, classes)

                          fmt = '.2f' if normalize else 'd'
                          thresh = cm.max() / 2.
                          for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                              plt.text(j, i, format(cm[i, j], fmt),
                                     horizontalalignment="center",
                                     color="white" if cm[i, j] > thresh else "black")

                          plt.tight_layout()
                          plt.ylabel('True label')
                          plt.xlabel('Predicted label')
