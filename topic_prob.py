#####################################################################
# Copyright(C), 2023 IHX Private Limited. All Rights Reserved
# Unauthorized copying of this file, via any medium is 
# strictly prohibited 
#
# Proprietary and confidential
# email: care@ihx.in
#####################################################################

import copy
from itertools import combinations, product
import numpy as np
import pandas as pd

class TopicProb():
    def __init__(self, df, topic_list, topic_field='topic',):
        self.df = df.copy(deep=True)
        self.topic_field = topic_field
        self.topic_list = topic_list
        self.ntopics = len(self.topic_list)
        self.field_list = None
        self.field_combination_list = None
        self.default_value = 'ALL'
        print(f'Length of df before dropping selected topics: {len(self.df)}')
        self.df = self.df[self.df[self.topic_field].isin(self.topic_list)]
        print(f'Length of df after dropping selected topics: {len(self.df)}') 
    
    def select_fields(self, field_list):
        self.field_list = list(field_list)
        if (len(self.field_list) == 0):
            # select the entire df
            pass
        
        # change the field values to str
        self.df = self.df.astype({field:str for field in self.field_list})
        
        # drop fields other than topic and the selected ones
        retain_field_list = self.field_list + [self.topic_field]
        remove_field_list = []
        for field in self.df.columns:
            if (field not in retain_field_list):
                remove_field_list.append(field)
        self.df.drop(columns=remove_field_list, inplace=True)
        print(f'dropped {len(remove_field_list)} fields')

        # set default field values
        self.default_field_values = {key: self.default_value for key in self.field_list}
    
    def __get_field_combinations(self):
        if (self.field_list is None):
            raise ValueError('Supply the required fields before calling this function uisng `select_fields` method')
        
        # obtain all comibnations
        self.field_combination_list = []
        for i in range(0, len(self.field_list)+1):
            self.field_combination_list = self.field_combination_list + list(combinations(self.field_list, i))
    
    
    def __get_filter_values(self, selected_fields):
        # get all possible combinations of fields
        return [self.df[field].unique().tolist() for field in selected_fields]

    def __get_number_of_filters(self):
        alist = []
        self.filter_size_list = [self.df[field].nunique() for field in self.field_list]
        for i in range(0, len(self.filter_size_list)+1):
            alist += list(combinations(self.filter_size_list, i))
        self.number_of_filters = int(np.sum([np.prod(np.array(l)) for l in alist]))
        
    def __calc_topic_freq(self):
        # calculate the topic counts for the
        # given set of filters
        pass

    def get_values_dict(self, field_combination, values):
        adict = copy.deepcopy(self.default_field_values)
        for f, v in zip(list(field_combination), list(values)):
            adict[f] = v
        return adict
    
    @staticmethod
    def query_string(field_combination, values):
        return ' & '.join(f'{field} == "{value}"' for field, value in zip(field_combination, values))

    def run_filters(self):
        # 0. Get all the filter values
        self.__get_number_of_filters()
        print(f'Total number of rows in the resulting table: {self.number_of_filters}')
        # 1. Get all the possible combinations of filters
        self.__get_field_combinations()

        alist = []
        counter = 0
        for selected_fields in self.field_combination_list[1:]:
            values_combination = product(*self.__get_filter_values(selected_fields))
            for values in values_combination:
                adict = self.get_values_dict(selected_fields, values)
                query = self.query_string(selected_fields, values)
                extracted_df = self.df.query(query)
                adict['length'] = len(extracted_df)
                # adict['df'] = extracted_df
                if (adict['length'] > 0):
                    topic_dist_pd = extracted_df[self.topic_field].value_counts(normalize=True)
                    topic_dist = np.zeros(self.ntopics)
                    topic_dist[topic_dist_pd.index] = topic_dist_pd.values
                    adict['TopicProbability'] = topic_dist
                    alist.append(adict)
                counter += 1
            print(selected_fields, [len(l) for l in values_combination], values_combination)
        print("counter", counter)

        # repeat the code for all ALL
        
        
        self.topic_probability_df = pd.DataFrame(alist)
            
