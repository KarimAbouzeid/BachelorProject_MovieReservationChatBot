"""
Created on May 17, 2016

@author: xiul, t-zalipt
"""

import json
from deep_dialog import dialog_config


class DialogManagerClient:
    """ A dialog manager to mediate the interaction between an agent and a customer """
    
    def __init__(self, user):
        self.user = user

        self.user_action = None
        self.reward = 0
        self.episode_over = False

    def initialize_episode(self):
        """ Refresh state for new dialog """
        
        self.reward = 0
        self.episode_over = False

        self.user_action = self.user.initialize_episode()
        actionOfUser = self.user_action
        
        if dialog_config.run_mode < 3:
            print ("New episode, user goal:")
            print (json.dumps(self.user.goal, indent=2))
            print(dialog_config.run_mode)
            self.print_function(user_action = self.user_action)

        return actionOfUser
    
    def next_turn(self, historyDictionary, record_training_data=True, ):
        """ This function initiates each subsequent exchange between agent and user (agent first) """
    
        ########################################################################
        #   CALL USER TO TAKE HER TURN
        ########################################################################
        self.sys_action = historyDictionary
        self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
        self.reward = self.reward_function(dialog_status)

         ########################################################################
        #   Update state tracker with latest user action
        ########################################################################
        # if self.episode_over != True:
        #     self.print_function(user_action = self.user_action)
        
        return (self.user_action,self.episode_over, self.reward)
    
    
 
    def reward_function(self, dialog_status):
        """ Reward Function 1: a reward function based on the dialog_status """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = -self.user.max_turn #10
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = 2*self.user.max_turn #20
        else:
            reward = -1
        return reward


    
    def print_function(self, user_action=None):
        """ Print Function """
        if user_action:
            if dialog_config.run_mode == 0:
                print ("Turn %d usr: %s" % (user_action['turn'], user_action['nl']))
            elif dialog_config.run_mode == 1: 
                print ("Turn %s usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots']))
            elif dialog_config.run_mode == 2: # debug mode, show both
                print ("Turn %d usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots']))
                print ("Turn %d usr: %s" % (user_action['turn'], user_action['nl']))
            
            # if self.agent.__class__.__name__ == 'AgentCmd': # command line agent
            #     user_request_slots = user_action['request_slots']
            #     if 'ticket'in user_request_slots.keys(): del user_request_slots['ticket']
            #     if len(user_request_slots) > 0:
            #         possible_values = self.state_tracker.get_suggest_slots_values(user_action['request_slots'])
            #         for slot in possible_values.keys():
            #             if len(possible_values[slot]) > 0:
            #                 print('(Suggested Values: %s: %s)' % (slot, possible_values[slot]))
            #             elif len(possible_values[slot]) == 0:
            #                 print('(Suggested Values: there is no available %s)' % (slot))
            #     else:
            #         kb_results = self.state_tracker.get_current_kb_results()
            #         print ('(Number of movies in KB satisfying current constraints: %s)' % len(kb_results))
            
