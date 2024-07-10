"""
Created on May 17, 2016

@author: xiul, t-zalipt
"""

import json
from . import StateTracker
from deep_dialog import dialog_config


class DialogManagerServer:
    """ A dialog manager to mediate the interaction between an agent and a customer """
    
    def __init__(self, agent, act_set, slot_set, movie_dictionary):
        self.agent = agent
        self.act_set = act_set
        self.slot_set = slot_set
        self.state_tracker = StateTracker(act_set, slot_set, movie_dictionary)
        self.user_action = None
        self.reward = 0
        self.episode_over = False

    def initialize_episode(self, actionOfUser):
        """ Refresh state for new dialog """
        
        self.reward = 0
        self.episode_over = False
        self.state_tracker.initialize_episode()
        self.user_action = actionOfUser
        self.state_tracker.update(user_action = self.user_action)
        self.agent.initialize_episode()

    
    def next_turn(self, record_training_data=True):
        """ This function initiates each subsequent exchange between agent and user (agent first) """
        
        ########################################################################
        #   CALL AGENT TO TAKE HER TURN
        ########################################################################
        self.state = self.state_tracker.get_state_for_agent()
        self.agent_action = self.agent.state_to_action(self.state)
        
        ########################################################################
        #   Register AGENT action with the state_tracker
        ########################################################################
        self.state_tracker.update(agent_action=self.agent_action)
        
        self.agent.add_nl_to_action(self.agent_action) # add NL to Agent Dia_Act
        self.print_function(agent_action = self.agent_action['act_slot_response'])
        
        return self.state_tracker.dialog_history_dictionaries()[-1]
    

    def next_turnCont(self, userAction, epiOver, rewardo,  record_training_data=True):

        if epiOver != True:
            self.state_tracker.update(user_action = userAction)
        ########################################################################
        #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over)
        ########################################################################
        if record_training_data:
            self.agent.register_experience_replay_tuple(self.state, self.agent_action, rewardo, self.state_tracker.get_state_for_agent(), self.episode_over) 



    def print_function(self, agent_action=None, user_action=None):
        """ Print Function """
            
        if agent_action:
            if dialog_config.run_mode == 0:
                if self.agent.__class__.__name__ != 'AgentCmd':
                    print ("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl']))
            elif dialog_config.run_mode == 1:
                if self.agent.__class__.__name__ != 'AgentCmd':
                    print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots']))
            elif dialog_config.run_mode == 2: # debug mode
                print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots']))
                print ("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl']))
            
            # if dialog_config.auto_suggest == 1:
            #     print('(Suggested Values: %s)' % (self.state_tracker.get_suggest_slots_values(agent_action['request_slots'])))