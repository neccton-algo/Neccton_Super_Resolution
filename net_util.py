import tensorflow as tf
import os
import pickle
import json
import numpy as np

def save_model_parameters(model_and_compile_params, model_history, paths):
    file_model_parameters = os.path.join(paths["outputs"],f"Model_parameters.txt")
    file_model_training_history = os.path.join(paths["outputs"],f"Training_history.pkl")
    #
    if os.path.isfile(file_model_parameters) == True:
        os.system("rm " + file_model_parameters)
    if os.path.isfile(file_model_training_history) == True:
        os.system("rm " + file_model_training_history)
    #
    pickle.dump(model_history.history, open(file_model_training_history, "wb"))
    with open(file_model_parameters, "w") as output_file:
        output_file.write(json.dumps(model_and_compile_params))