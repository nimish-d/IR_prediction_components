#####################################################################
# Copyright(C), 2023 IHX Private Limited. All Rights Reserved
# Unauthorized copying of this file, via any medium is 
# strictly prohibited 
#
# Proprietary and confidential
# email: care@ihx.in
#####################################################################

import os
import pickle
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

    def __init__(self):
        self.representation_loaded_from = None
        self.bertopic_model_path = None
        self.representation_dict_pickle_path = None
        self.ctfidf_selection = None
        self.action_topic_probability_matrix_path = None

    @staticmethod
    def softmax2d(anarray):
        '''
        argument: a 2d array of numbers
        returns: softmax probability distribution.
                 The calculation is performed column-wise.
        '''
        exp_array = np.exp(anarray)
        normalization_array = np.sum(exp_array, axis=0)
        return exp_array / normalization_array[:]

    def read_bertopic_model(self, bertopic_model_path):
        self.bertopic_model_path = bertopic_model_path
        self.representation_loaded_from = 'bertopic'
        # Load a bertopic model
        self.topic_model = BERTopic.load(self.bertopic_model_path)

        # get the dictionary of representation (value) for each topic (key)
        # representation is a list of tuples of size nreps
        # each tuple contains the representative string and the corresponding ctfidf value
        self.representation_dict = self.topic_model.topic_representations_

        # get parameters of the topic model
        self.topic_param_dict = self.topic_model.get_params()

        # get the number of representations (nreps) per topic
        self.nreps = int(self.topic_param_dict['top_n_words'])
        self.rep_list = list(range(self.nreps))
        self.ntopics = len(self.representation_dict) - 1 # exclude topic number '-1'
        self.topic_list = list(range(self.ntopics))
    
    def read_representation_dictionary(self, representation_dict_pickle_path):
        self.representation_dict_pickle_path = representation_dict_pickle_path
        self.representation_loaded_from = 'representation dictionary pickle'
        with open(self.representation_dict_pickle_path ,'rb') as handle:
            self.representation_dict = pickle.load(handle)
        handle.close()

        # nreps counted from the representations of the first topic
        # assuming number of reps don't change with topic
        self.nreps = len(self.representation_dict[0])
        self.rep_list = list(range(self.nreps))
        self.ntopics = len(self.representation_dict) - 1 # exclude topic number '-1'
        # assuming topic idx is contiguous between 0 to ntopics
        self.topic_list = list(range(self.ntopics))

    def read_rep_action_mapping_csv(self, rep_action_mapping_csv_path, nactions=None):
        self.rep_action_mapping_csv_path = rep_action_mapping_csv_path
        self.action_representation_map_df = pd.read_csv(self.rep_action_mapping_csv_path)
        self.action_list = self.action_representation_map_df['ActionID'].unique()

        if (nactions is None):
            self.nactions = len(self.action_list)
        else:
            self.nactions = nactions

        # 3d array containing the mapping between between actions and representations for every topic
        # shape: nactions x ntopics x nreps
        self.ctfidf_selection = np.zeros([self.nactions, self.ntopics, self.nreps])

        for i in range(len(self.action_representation_map_df)):
                adict = self.action_representation_map_df.iloc[i].to_dict()
                action_id = adict['ActionID']
                topic_number = adict['TopicNumber']
                unmask_indices = eval(adict['RepresentationList'])
                self.ctfidf_selection[action_id, topic_number, unmask_indices] = 1

    def read_additional_topic_actions_csv(self):
        pass

    def read_action_prob_file(self, action_topic_probabity_matrix_path):
        self.action_topic_probability_matrix_path = action_topic_probabity_matrix_path
        self.action_probability_topic_wise = np.load(self.action_topic_probability_matrix_path)

    def save_action_prob_file(self, action_topic_probabity_matrix_path):
        self.action_topic_probability_matrix_path = action_topic_probabity_matrix_path
        np.save(self.action_topic_probability_matrix_path, self.action_probability_topic_wise)
        print(f'Saved action probabilities in the path {self.action_topic_probability_matrix_path}')

    def __calc(self):
        # collect the ctfidf values of all the representation strings for each topic into
        # a 2d list; array shape: number of topics x number of representations per topic
        # note: topic number -1 is ignored
        self.ctfidf_topic_rep = np.zeros([self.ntopics, self.nreps])
        for topic_no in self.topic_list:
            self.ctfidf_topic_rep[topic_no, :] = np.array([float(self.representation_dict[topic_no][k][1]) for k in self.rep_list])
        
        # replicate ctfidf_topic_rep array naction times
        # shape: nactions x ntopics x nreps
        ctfidf_repeated_3d = np.repeat(self.ctfidf_topic_rep[np.newaxis, :, :], self.nactions, axis=0)

        # for each action, retain only ctfidfs of representations for every topic defined by the ctfidf_selection
        # element-wise multiplication (selection) ctfidf_selectiona and ctfidf_repeated_3d
        self.action_topic_wise_ctfidf = self.ctfidf_selection * ctfidf_repeated_3d
        del ctfidf_repeated_3d

        # calculate the sum of the selected ctfidf values for selected representations for each action for each topic
        # shape: nactions x ntopics
        self.action_topic_wise_ctfidf_sum = np.sum(self.action_topic_wise_ctfidf, axis=2)

        # calculated softmax action probabilities from the ctfidf sums for each topic
        # shape: nactions x ntopics
        self.action_probability_topic_wise = self.softmax2d(self.action_topic_wise_ctfidf_sum)

    def calc_action_probs(self):
        # 1. Check if representation dict exists
        if (self.bertopic_model_path is None
            and self.representation_dict_pickle_path is None):
            print(f'[Error] representation dictionary does not exist. Load a bertopic model or representation_dict before invoking calc_action_probs method')
            return None

        # 2. Check if rep action mapping is present
        if (self.ctfidf_selection is None):
            print(f'[Error] representation-action mapping not present.')
            return None

        # 3 (optional). Read additional actions csv
        self.read_additional_topic_actions_csv()

        # 4. Calculate action probabilities for every topic
        self.__calc()