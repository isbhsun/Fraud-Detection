import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, cross_val_score
from sklearn.metrics import recall_score, precision_score, f1_score
import time

def get_data():
    # this is just code from the eda notebook to set up the X_train, y_train, X_test, y_test
    import pandas as pd
    import zipfile
    zf = zipfile.ZipFile('data/data.zip')
    df = pd.read_json(zf.open('data.json'))
    df['fraud'] = df['acct_type'].apply(lambda x: True if 'fraud' in x else False)
    drop_cols = ['has_header', 
             'venue_address',
             'venue_country',
             'venue_latitude',
             'venue_longitude',
             'venue_name',
             'venue_state',
             'sale_duration']
    for col in drop_cols:
        try:
            df.pop(col)
        except:
            continue
    look_later = ['acct_type',
                  'description',
                  'name',
                  'org_desc',
                  'org_name',
                  'payee_name',
                  'previous_payouts',
                  'ticket_types', # grab cost out of ticket types
                  'user_created',
                  'email_domain'] # maybe create dummies for anonymous email domains
    for col in look_later:
        try:
            df.pop(col)
        except:
            continue
    df.dropna(inplace=True)

    df['listed'] = df['listed'].apply(lambda x: 1 if 'y' else 0)
    df_analysis = pd.get_dummies(df, prefix_sep='_')

    return df_analysis


def gridsearch_with_output(estimator, parameter_grid, score, X_train, y_train):
    '''
        Parameters: estimator: the type of model (e.g. RandomForestRegressor())
                    paramter_grid: dictionary defining the gridsearch parameters
                    score: string or a callable to evalute the predictions
                    X_train: 2d numpy array
                    y_train: 1d numpy array
        Returns:  best parameters and model fit with those parameters
                  Also returns the best score
    '''
    model_gridsearch = GridSearchCV(estimator,
                                    parameter_grid,
                                    n_jobs=-1,
                                    verbose=True,
                                    scoring=score)
    model_gridsearch.fit(X_train, y_train)
    best_params = model_gridsearch.best_params_ 
    model_best = model_gridsearch.best_estimator_
    best_score = model_gridsearch.best_score_
    print("\nResult of gridsearch:")
    print("{0:<20s} | {1:<8s} | {2}".format("Parameter", "Optimal", "Gridsearch values"))
    print("-" * 55)
    for param, vals in parameter_grid.items():
        print("{0:<20s} | {1:<8s} | {2}".format(str(param), 
                                                str(best_params[param]),
                                                str(vals)))
    return best_params, model_best, best_score

def score_classifer(model, model_name, X_train, y_train):
    start_time = time.time()
    cv_scores = cross_validate(model, X_train, y_train,
                           cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=1), 
                           return_train_score=False, 
                           scoring=["recall", "precision", "f1"])
    
    cv_Recall_scores = cv_scores['test_recall']
    cv_Precision_scores = cv_scores['test_precision']         
    cv_f1_scores = cv_scores['test_f1']   
    
    # use average to calculate a singel score:
    cvAvg_Recall_score = np.mean(cv_Recall_scores)
    cvAvg_Precision_score = np.mean(cv_Precision_scores)
    cvAvg_f1_score = np.mean(cv_f1_scores)

    print(f"{time.time() - start_time:.0f} seconds cvfit execution time for {model_name}")
    print(f'Recall: {cvAvg_Recall_score}')
    print(f'Precision: {cvAvg_Precision_score}')
    print(f'f1_score: {cvAvg_f1_score}')
    return cvAvg_Recall_score, cvAvg_Precision_score, cvAvg_f1_score

def test_classifer(model, model_name, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    test_Recall_score = recall_score(y_test, y_predict)
    test_Precision_score = precision_score(y_test, y_predict) 
    test_f1_score = f1_score(y_test, y_predict)  
    
    print(f"{time.time() - start_time:.0f} seconds execution time for {model_name}")
    print(f'Recall: {test_Recall_score}')
    print(f'Precision: {test_Precision_score}')
    print(f'f1_score: {test_f1_score}')
    return test_Recall_score, test_Precision_score, test_f1_score

if __name__ == '__main__':
    get_data()

    # instantiate classifiers
    rf = RandomForestClassifier(n_jobs=-1, random_state=0)
    xgb = XGBClassifier(n_jobs=-1, random_state=1)
    gbc = GradientBoostingClassifier(random_state=1)

    # obtain cross validation scores
    r,p,f =score_classifer(rf, "random forest classifier")
    r,p,f =score_classifer(xgb, "xgboost classifier")
    r,p,f =score_classifer(gbc, "gradient boost classifier")

    # obtain scores using test dataset
    r,p,f = test_classifer(rf, "random forest classier")
    r,p,f = test_classifer(xgb, "xgboost classifier")
    r,p,f = test_classifer(gbc, "gradient boost classifier")

    # instantiate voting classifier after other estimators have already been fit
    estimators = [['rf',rf], ['xgb', xgb], ['gbc', gbc]]
    voting = VotingClassifier(estimators, voting = 'soft')
    r,p,f = test_classifer(voting, "bagging ensemble voting classifier")

