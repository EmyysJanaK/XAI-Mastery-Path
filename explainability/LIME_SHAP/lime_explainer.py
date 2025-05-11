'''
     Explains predictions by approximating the locally interpretable model.
'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular 

data = load_iris()
X, y = data.data, data.target

rf = RandomForestClassifier(n_estimators=100).fit(X, y)

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train=X, feature_names=data.feature_names, class_names=data.target_names, discretize_continuous=True
)

i = 0  # pick an instance
exp = explainer.explain_instance(X[i], rf.predict_proba, num_features=4)
exp.show_in_notebook()

def explain_with_lime(model, X, feature_names, class_names, num_features=5):

    """
    Explain the predictions of a model using LIME.  
    Args:
        model: The trained model to explain.
        X: The input data to explain.
        feature_names: Names of the features in the dataset.
        class_names: Names of the classes in the dataset.
        num_features: The number of features to include in the explanation.

    Returns:
        A LIME explanation object.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    
    return explainer


