#####################################################################
# Copyright(C), 2023 IHX Private Limited. All Rights Reserved
# Unauthorized copying of this file, via any medium is 
# strictly prohibited 
#
# Proprietary and confidential
# email: care@ihx.in
#####################################################################

import path
import yaml
import numpy as np
import pandas as pd
from bertopic import BERTopic

class ActionProb():
    '''
    Each topic of Historical IR remarks suggests one or more actions. The importance of
    each action is calculated and returned by this class. cTF-IDF values of the string-representions
    of the topic are combined and then a softmax function is applied.
    Author: Nimish Dwarkanath
    '''

    def __init__(self, action_prob_path):
        self.action_prob_path = action_prob_path
        # check if a file with action weights exists
        # if not read the model (pytensors)


    def read_bertopic_model(self):
    def read_rep_action_mapping_yaml(self):
    def read_action_prob_file(self):
    def save_action_prob_file(self):
    def remove_action_prob_file(self):
    def __calc(self):
        pass
    def calc_action_probs(self, bertopic_model_path, rep_action_mapping_yaml_path, overwrite=False):
        # If overwrite is False and the action_prob_file exists --> read and skip the calculation
        # If overwrite is True and the action_prob_file exists --> remove the file
        # else calculate the action probability
            # 1. Load the bertopic model
            # 2. Check if the ctfidf scores are recorded, if not throw an error and quit
            # 3. Read the representation - action mapping file (YAML)
            # 4. Calculate action probabilities for every topic
        # return action probs



# Read the ctfidf weights from the model
# 

