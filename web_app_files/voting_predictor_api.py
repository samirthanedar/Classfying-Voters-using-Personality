"""
Note this file contains _NO_ flask functionality.
Instead it makes a file that takes the input dictionary Flask gives us,
and returns the desired result.

This allows us to test if our modeling is working, without having to worry
about whether Flask is working. A short check is run at the bottom of the file.
"""

import pickle
import numpy as np


with open("flask_rf.pkl", "rb") as f:
    rf_model = pickle.load(f)

feature_names = ['What year were you born?', 'Are you a male or female?', 
'Does your life have a purpose', 'Do you meditate or pray on a daily basis?',
'Would you say most of the hardship in your life has been the result of circumstances beyond your own control, or has it been mostly the result of your own decisions and actions?',
'Which parent "wore the pants" in your household?', 'Would you rather be happy or right?',
'Do you personally own a gun?', 'Do you live within 20 miles of a major metropolitan area?',
'Are you a feminist?', 'Are you more of an idealist or a pragmatist?', 'Did your parents spank you as a form of discipline/punishment?']

target_names = ['Democrat', 'Republican']


def make_prediction(feature_dict):
    """
    Input:
    feature_dict: a dictionary of the form {"feature_name": "value"}

    Function makes sure the features are fed to the model in the same order the
    model expects them.

    Output:
    Returns (x_inputs, probs) where
      x_inputs: a list of feature values in the order they appear in the model
      probs: a list of dictionaries with keys 'name', 'prob'
    """
    #feature_names = feature_dict.keys()

    x_input = [
        (float(feature_dict.get(name, 0))) for name in feature_names
    ]

    pred_probs = rf_model.predict_proba([x_input]).flat

    probs = [{'name': target_names[index], 'prob': pred_probs[index]}
             for index in np.argsort(pred_probs)[::-1]]
    return x_input, probs


# This section checks that the prediction code runs properly
# To run, type "python predictor_api.py" in the terminal.
#
# The if __name__='__main__' section ensures this code only runs
# when running this file; it doesn't run when importing
#if __name__ == '__main__':
    #from pprint import pprint
    #print("Checking to see what setting all params to 0 predicts")
    #features = {f: '0' for f in feature_names}
    #print('Features are')
    #pprint(features)

    #x_input, probs = make_prediction(features)
    #print(f'Input values: {x_input}')
    #print('Output probabilities')
    #pprint(probs)
