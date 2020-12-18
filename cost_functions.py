import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def cs_confusion_matrix(y_test, y_pred, cost_matrix):
    """
    Returns a cost sensitive confusion matrix using the cost matrix
 
    Parameters
    ----------
    y_test: array 
    y_pred: array
    cost_matrix: array

    Returns
    -------    
    array
    """

    cost_TN = np.sum((1 - y_test) * (1 - y_pred) * cost_matrix[:, 0])
    cost_FP = np.sum((1 - y_test) * y_pred * cost_matrix[:, 1])
    cost_FN = np.sum(y_test * (1 - y_pred) * cost_matrix[:, 2])
    cost_TP = np.sum(y_test * y_pred * cost_matrix[:, 3])

    return np.array([[cost_TN, cost_FP],[cost_FN, cost_TP]])

def cost_matrix(c):
    '''
    Returns a cost_matrix that contains, for every transaction, the four 
    possible costs associated with it depending on the outcome of its 
    classification (TN, FP, FN, TP). 

    Parameters
    ----------
    c: array consisting of each transaction's dollar amount

    Returns
    -------
    cost_matrix: array
    ''' 

    n_samples = c.shape[0]
    cost_matrix = np.zeros((n_samples, 4))
    cost_matrix[:, 0] = c #0.0
    cost_matrix[:, 1] = c                                                               
    cost_matrix[:, 2] = c 
    cost_matrix[:, 3] = c # 0.0
    return cost_matrix 



def plot_confusion_matrix(ax, cm, title, classes=['Legitimate','Fraud'],
                          cmap=plt.cm.Blues, currency=False):
    """
    Plots a single confusion matrix. If currency=True then displays results as currency.

    Parameters
    ----------
    cm: array (confusion matrix)
    title: String
    test_size: float - size/percentage of holout dataset
    goal: float - project goal for ultimate dollar loss rate

    Returns
    -------
    """   
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        cost=cm[i, j]
        if currency:
            cost = f'${cost:0,.2f}' 
        ax.text(j, i, cost, horizontalalignment="center", 
        color="white" if cm[i, j] > thresh else "black")
    ax.imshow(cm, interpolation='nearest', cmap=cmap)

    if currency:
        ax.set_title(f'{title}\nCost Matrix')
    else:
        ax.set_title(f'{title}\nConfusion Matrix')
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=0)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, rotation=90)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')    


def print_confusion(model, title):
    model.fit(X_train, y_train)
    pred_prob = model.predict_proba(X_test)[:,1]
    threshold = 0.5

    # print confusion matrix
    cnf_matrix = confusion_matrix(y_test, pred_prob >= threshold)
    fix, ax = plt.subplots()
    plot_confusion_matrix(ax, cnf_matrix, title, classes=['Legitimate','Fraud'],
                              cmap=plt.cm.Blues, currency=False)

    # print cost matrix
    cs_cnf_matrix = cs_confusion_matrix(y_test, pred_prob >= threshold, cost_matrix(c_test)) 
    fix, ax = plt.subplots()
    plot_confusion_matrix(ax, cs_cnf_matrix, title, classes=['Legitimate','Fraud'],
                          cmap=plt.cm.Blues, currency=True)

if __name__ == '__main__':
    rf = RandomForestClassifier(n_jobs=-1,
                            max_depth=2,
                            max_features='sqrt',
                            oob_score=True,
                            n_estimators=20,
                            class_weight='balanced',
                            random_state=0)
    model = rf
    title = "Random Forest Classifier"
    print_confusion(model, title)