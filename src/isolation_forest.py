from sklearn.ensemble import IsolationForest
from cost_functions import plot_confusion_matrix
from cost_functions import cs_confusion_matrix
from cost_functions import cost_matrix

def train(X,clf,ensembleSize=5,sampleSize=10000):
    mdlLst=[]
    for n in range(ensembleSize):
        X=df_analysis.sample(sampleSize)
        clf.fit(X)
        mdlLst.append(clf)
    return mdlLst

def predict(X,mdlLst):
    y_pred=np.zeros(X.shape[0])
    for clf in mdlLst:
        y_pred=np.add(y_pred,clf.decision_function(X).reshape(X.shape[0],))
    y_pred=(y_pred*1.0)/len(mdlLst)
    return y_pred

def print_isf_confusion(model, title):
    isf=IsolationForest(n_estimators=100, max_samples='auto', contamination=0.01, \
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0,behaviour="new")

    # fit the classifier
    mdlLst=train(X_train,isf)

    # predict with test data
    y_pred=predict(X_test, mdlLst)
    y_pred=1-y_pred

    y_pred_class=y_pred.copy()
    y_pred_class[y_pred>=np.percentile(y_pred,95)]=1
    y_pred_class[y_pred<np.percentile(y_pred,95)]=0

    # print confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred_class)
    fix, ax = plt.subplots()
    plot_confusion_matrix(ax, cnf_matrix, title, classes=['Legitimate','Fraud'],
                              cmap=plt.cm.Blues, currency=False)

    # print cost matrix
    cs_cnf_matrix = cs_confusion_matrix(y_test, y_pred_class, cost_matrix(c_test)) 
    fix, ax = plt.subplots()
    plot_confusion_matrix(ax, cs_cnf_matrix, title, classes=['Legitimate','Fraud'],
                          cmap=plt.cm.Blues, currency=True)

def isolation_forest_test_classifier():
        isf=IsolationForest(n_estimators=100, max_samples='auto', contamination=0.01, \
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0,behaviour="new")
    # fit the classifier
    mdlLst=train(X_train,isf)

    # predict with test data
    y_pred=predict(X_test, mdlLst)
    y_pred=1-y_pred

    y_pred_class=y_pred.copy()
    y_pred_class[y_pred>=np.percentile(y_pred,95)]=1
    y_pred_class[y_pred<np.percentile(y_pred,95)]=0

    p, r, f = precision_recall_fscore_support(y_test, y_pred_class, average='weighted')
    print(f" model_name}")
    print(f'Recall: {r}')
    print(f'Precision: {p}')
    print(f'f1_score: {f}')
    return p, r, f