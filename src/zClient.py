import socket

import argparse, json, copy, os
import pickle

from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.agents import AgentCmd, InformAgent, RequestAllAgent, RandomAgent, EchoAgent, RequestBasicsAgent, AgentDQN
from deep_dialog.usersims import RuleSimulator

from deep_dialog.dialog_system import DialogManager, DialogManagerClient, DialogManagerServer, text_to_dict
from deep_dialog.dialog_config import *

import datetime as dt #Added to count time

from deep_dialog import dialog_config
from deep_dialog.dialog_config import *

from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg
import gzip


HEADER = 64
PORT = 5555
DISCONNECT_MESSAGE = "!DISCONNECT"
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)


class zClient:


    def send(self, message, type):
        msg = message
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(ADDR)
        
        if (type == 'params'):
            
            msg = msg.encode()
            client.sendall(b'params'+msg)

        elif(type == 'initialize'):
            
            client.sendall(bytes('initia', "utf-8")+msg)
            

        elif (type == 'nextturn1'):
            
            client.sendall(bytes('nxtrn1', "utf-8")+msg)
            response = client.recv(2048)
            response = pickle.loads(response)
            return (response)
        
        elif (type == 'nextturn2'):
            
            client.sendall(bytes('nxtrn2', "utf-8")+msg)

        elif (type == 'cumulative_turns'):
            
            client.sendall(bytes('cumtrn', "utf-8")+msg)
            return (client.recv(2048).decode())
        
        elif (type == 'connect'):
            
            client.sendall(bytes('connec', "utf-8")+msg)

        elif (type == 'disconnect'):
            client.sendall(bytes('discon', "utf-8")+msg)

        elif (type == 'experience_replay_pool'):
            client.sendall(bytes('exprpo', "utf-8")+msg)
            return (client.recv(2048).decode())
        
        elif (type == 'experience_replay_pool_size'):
            client.sendall(bytes('exprps', "utf-8")+msg)
            return (client.recv(2048).decode())
        
        elif (type == 'agent_warm_start'):
            client.sendall(bytes('agwrst', "utf-8")+msg)

        elif (type == 'agent_predict_mode_True'):
            client.sendall(bytes('agprmt', "utf-8")+msg)

        elif (type == 'agent_experience_replay_pool_reset'):
            client.sendall(bytes('agrpre', "utf-8")+msg)

        elif (type == 'best_model_update'):
            client.sendall(bytes('bemdup', "utf-8")+msg)
        
        elif (type == 'agent_predict_mode_False'):
            client.sendall(bytes('agprmf', "utf-8")+msg)

        elif (type == 'agent_clone_dqn'):
            client.sendall(bytes('agcldq', "utf-8")+msg)

        elif (type == 'agent_train'):
            client.sendall(bytes('agtrbs', "utf-8")+msg)

        
        # elif (type == 'agent_copy'):
        #     client.sendall(bytes('agcopy', "utf-8")+msg)

            # response = client.recv(2048)
            
            # response = pickle.loads(response)
            # return (response)
        
        elif (type == 'save_model'):
            client.sendall(bytes('savmod', "utf-8")+msg)

        elif (type == 'save_performance_records'):
            client.sendall(bytes('savper', "utf-8")+msg)

        elif (type == 'save_model_2'):
            client.sendall(bytes('savmd2', "utf-8")+msg)
        
    
        
    

if __name__ == "__main__":
    clientClass = zClient()


    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_path", dest="dict_path", type=str, default="./deep_dialog/data/dicts.v3.p", help="path to the json dictionary file")
    parser.add_argument("--movie_kb_path", dest="movie_kb_path", type=str, default="./deep_dialog/data/movie_kb.1k.p", help="path to the movie kb .json file")
    parser.add_argument("--act_set", dest="act_set", type=str, default="./deep_dialog/data/dia_acts.txt", help="path to dia act set; none for loading from labeled file")
    parser.add_argument("--slot_set", dest="slot_set", type=str, default="./deep_dialog/data/slot_set.txt", help="path to slot set; none for loading from labeled file")
    parser.add_argument("--goal_file_path", dest="goal_file_path", type=str, default="./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p", help="a list of user goals")
    parser.add_argument("--diaact_nl_pairs", dest="diaact_nl_pairs", type=str, default="./deep_dialog/data/dia_act_nl_pairs.v6.json", help="path to the pre-defined dia_act&NL pairs")

    parser.add_argument("--max_turn", dest="max_turn", default=20, type=int, help="maximum length of each dialog (default=20, 0=no maximum length)")
    parser.add_argument("--episodes", dest="episodes", default=1, type=int, help="Total number of episodes to run (default=1)")
    parser.add_argument("--slot_err_prob", dest="slot_err_prob", default=0.05, type=float, help="the slot err probability")
    parser.add_argument("--slot_err_mode", dest="slot_err_mode", default=0, type=int, help="slot_err_mode: 0 for slot_val only; 1 for three errs")
    parser.add_argument("--intent_err_prob", dest="intent_err_prob", default=0.05, type=float, help="the intent err probability")
    
    parser.add_argument("--agt", dest="agt", default=9, type=int, help="Select an agent: 0 for a command line input, 1-6 for rule based agents")
    parser.add_argument("--usr", dest="usr", default=1, type=int, help="Select a user simulator. 0 is a Frozen user simulator.")
    
    parser.add_argument("--epsilon", dest="epsilon", type=float, default=0, help="Epsilon to determine stochasticity of epsilon-greedy agent policies")
    
    # load NLG & NLU model
    parser.add_argument("--nlg_model_path", dest="nlg_model_path", type=str, default="./deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p", help="path to model file")
    parser.add_argument("--nlu_model_path", dest="nlu_model_path", type=str, default="./deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p", help="path to the NLU model file")
    
    parser.add_argument("--act_level", dest="act_level", type=int, default=0, help="0 for dia_act level; 1 for NL level")
    parser.add_argument("--run_mode", dest="run_mode", type=int, default=3, help="run_mode: 0 for default NL; 1 for dia_act; 2 for both")
    parser.add_argument("--auto_suggest", dest="auto_suggest", type=int, default=0, help="0 for no auto_suggest; 1 for auto_suggest")
    parser.add_argument("--cmd_input_mode", dest="cmd_input_mode", type=int, default=0, help="run_mode: 0 for NL; 1 for dia_act")
    
    # RL agent parameters
    parser.add_argument("--experience_replay_pool_size", dest="experience_replay_pool_size", type=int, default=1000, help="the size for experience replay")
    parser.add_argument("--dqn_hidden_size", dest="dqn_hidden_size", type=int, default=60, help="the hidden size for DQN")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--gamma", dest="gamma", type=float, default=0.9, help="gamma for DQN")
    parser.add_argument("--predict_mode", dest="predict_mode", type=bool, default=False, help="predict model for DQN")
    parser.add_argument("--simulation_epoch_size", dest="simulation_epoch_size", type=int, default=50, help="the size of validation set")
    parser.add_argument("--warm_start", dest="warm_start", type=int, default=1, help="0: no warm start; 1: warm start for training")
    parser.add_argument("--warm_start_epochs", dest="warm_start_epochs", type=int, default=100, help="the number of epochs for warm start")
    
    parser.add_argument("--trained_model_path", dest="trained_model_path", type=str, default=None, help="the path for trained model")
    parser.add_argument("-o", "--write_model_dir", dest="write_model_dir", type=str, default="./deep_dialog/checkpoints/", help="write model to disk") 
    parser.add_argument("--save_check_point", dest="save_check_point", type=int, default=10, help="number of epochs for saving model")
     
    parser.add_argument("--success_rate_threshold", dest="success_rate_threshold", type=float, default=0.3, help="the threshold for success rate")
    
    parser.add_argument("--split_fold", dest="split_fold", default=5, type=int, help="the number of folders to split the user goal")
    parser.add_argument("--learning_phase", dest="learning_phase", default="all", type=str, help="train/test/all; default is all")
    
    args = parser.parse_args()
    params = vars(args)



    params_str = json.dumps(params, separators=(',', ':'))


  
    clientClass.send(params_str,"params")






    max_turn = params['max_turn']
    num_episodes = params['episodes']

    agt = params['agt']
    usr = params['usr']

    dict_path = params['dict_path']
    goal_file_path = params['goal_file_path']

    # load the user goals from .p file
    all_goal_set = pickle.load(open(goal_file_path, 'rb'))

    # split goal set
    split_fold = params.get('split_fold', 5)
    goal_set = {'train':[], 'valid':[], 'test':[], 'all':[]}

    for u_goal_id, u_goal in enumerate(all_goal_set):
        if u_goal_id % split_fold == 1: goal_set['test'].append(u_goal)
        else: goal_set['train'].append(u_goal)
        goal_set['all'].append(u_goal)
    # end split goal set

    movie_kb_path = params['movie_kb_path']
    with open(movie_kb_path, "rb") as f:
        movie_kb = pickle.load(f, encoding="latin1")

    # movie_kb = pickle.load(open(movie_kb_path, 'rb'))

    act_set = text_to_dict(params['act_set'])
    slot_set = text_to_dict(params['slot_set'])

    ################################################################################
    # a movie dictionary for user simulator - slot:possible values
    ################################################################################
    movie_dictionary = pickle.load(open(dict_path, 'rb'))





    ################################################################################
    #   Parameters for User Simulators
    ################################################################################
    usersim_params = {}
    usersim_params['max_turn'] = max_turn
    usersim_params['slot_err_probability'] = params['slot_err_prob']
    usersim_params['slot_err_mode'] = params['slot_err_mode']
    usersim_params['intent_err_probability'] = params['intent_err_prob']
    usersim_params['simulator_run_mode'] = params['run_mode']

    usersim_params['simulator_act_level'] = params['act_level']
    usersim_params['learning_phase'] = params['learning_phase']

    if usr == 0:# real user
        user_sim = RealUser(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
    elif usr == 1: 
        user_sim = RuleSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)


    ################################################################################
    # load trained NLG model
    ################################################################################
    nlg_model_path = params['nlg_model_path']
    diaact_nl_pairs = params['diaact_nl_pairs']
    nlg_model = nlg()
    nlg_model.load_nlg_model(nlg_model_path)
    nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs)

   #////# SOS agent.set_nlg_model(nlg_model)
    user_sim.set_nlg_model(nlg_model)


    ################################################################################
    # load trained NLU model
    ################################################################################
    nlu_model_path = params['nlu_model_path']
    nlu_model = nlu()
    nlu_model.load_nlu_model(nlu_model_path)

   #////# SOS agent.set_nlu_model(nlu_model)
    user_sim.set_nlu_model(nlu_model)


    dialog_config.run_mode = params['run_mode']
    dialog_config.auto_suggest = params['auto_suggest']
    ################################################################################
    # Dialog Manager
    ################################################################################

    dialog_managerClient = DialogManagerClient (user_sim)

   # dialog_managerServer = DialogManagerServer(agent, act_set, slot_set, movie_kb)

    status = {'successes': 0, 'count': 0, 'cumulative_reward': 0}
    simulation_epoch_size = params['simulation_epoch_size']
    batch_size = params['batch_size'] # default = 16
    warm_start = params['warm_start']
    warm_start_epochs = params['warm_start_epochs']

    success_rate_threshold = params['success_rate_threshold']
    save_check_point = params['save_check_point']


    """ Best Model and Performance Records """
   # best_model = {}
    best_res = {'success_rate': 0, 'ave_reward':float('-inf'), 'ave_turns': float('inf'), 'epoch':0}
    # agent = clientClass.send(pickle.dumps(""), "agent_copy")
   # best_model['model'] = copy.deepcopy(agent)
    best_res['success_rate'] = 0

    performance_records = {}
    time = {}

    
    performance_records['success_rate'] = {}
    performance_records['ave_turns'] = {}
    performance_records['ave_reward'] = {}
    time['time_taken'] = {}
    time['time_taken_so_far'] = {}
    time['time_taken_so_far_minutes'] = {}

    """ Save model """
    def save_model(path, agt, success_rate, agent, best_epoch, cur_epoch):
        filename = 'agt_%s_%s_%s_%.5f.p' % (agt, best_epoch, cur_epoch, success_rate)
        filepath = os.path.join(path, filename)
        checkpoint = {}
        if agt == 9: checkpoint['model'] = copy.deepcopy(agent.dqn.model)
        checkpoint['params'] = params
        try:
            pickle.dump(checkpoint, open(filepath, "wb"))
            print ('saved model in %s' % (filepath, ))
        except Exception as e:
            print ('Error: Writing model fails: %s' % (filepath, ))
            print (e)

    """ save performance numbers """
    def save_performance_records(path, agt, records):
        filename = 'agt_%s_performance_records.json' % (agt)
        filepath = os.path.join(path, filename)
        try:
            json.dump(records, open(filepath, "wb"))
            print ('saved model in %s' % (filepath, ))
        except Exception as e:
            print ('Error: Writing model fails: %s' % (filepath, ))
            print (e)

    """ Run N simulation Dialogues """
    def simulation_epoch(simulation_epoch_size):
        successes = 0
        cumulative_reward = 0
        cumulative_turns = 0
        
        res = {}
        for episode in range(simulation_epoch_size):
        #   dialog_manager.initialize_episode()

            actionOfUser = dialog_managerClient.initialize_episode()
            actionOfUser = pickle.dumps(actionOfUser)
            clientClass.send(actionOfUser, "initialize")

            episode_over = False
            while(not episode_over):
            #   episode_over, reward = dialog_manager.next_turn()

                historydictionary = clientClass.send(pickle.dumps("")  , "nextturn1")

                actionUser2,episode_over, reward = dialog_managerClient.next_turn(historydictionary)
                
                nextturn2Object = {"actionUser2": actionUser2, "episode_over": episode_over, "reward": reward}
                nextturn2ObjectStr = pickle.dumps(nextturn2Object)
                clientClass.send(nextturn2ObjectStr, "nextturn2")
                    
                cumulative_reward += reward
                if episode_over:
                    if reward > 0: 
                        successes += 1
                        #print ("simulation episode %s: Success" % (episode))
                    #else: print ("simulation episode %s: Fail" % (episode))
                    turnsCount = clientClass.send(pickle.dumps(""), "cumulative_turns")
                    turnsCount = int(turnsCount)
                    cumulative_turns+= turnsCount  
        
        res['success_rate'] = float(successes)/simulation_epoch_size
        res['ave_reward'] = float(cumulative_reward)/simulation_epoch_size
        res['ave_turns'] = float(cumulative_turns)/simulation_epoch_size
        print ("simulation success rate %s, ave reward %s, ave turns %s" % (res['success_rate'], res['ave_reward'], res['ave_turns']))
        return res

    """ Warm_Start Simulation (by Rule Policy) """
    def warm_start_simulation():
        successes = 0
        cumulative_reward = 0
        cumulative_turns = 0
        
        res = {}
        warm_start_run_epochs = 0
        for episode in range(warm_start_epochs):

            if(dialog_config.run_mode<3):
                print ("Episode: %s" % (episode))
            clientClass.send(pickle.dumps(episode), "connect")
       
            actionOfUser = dialog_managerClient.initialize_episode()
            actionOfUser = pickle.dumps(actionOfUser)
            clientClass.send(actionOfUser, "initialize")

            episode_over = False

            while(not episode_over):
            #   episode_over, reward = dialog_manager.next_turn()

                historydictionary = clientClass.send(pickle.dumps("")  , "nextturn1")

                actionUser2,episode_over, reward = dialog_managerClient.next_turn(historydictionary)
                
                nextturn2Object = {"actionUser2": actionUser2, "episode_over": episode_over, "reward": reward}
                nextturn2ObjectStr = pickle.dumps(nextturn2Object)
                clientClass.send(nextturn2ObjectStr, "nextturn2")

                cumulative_reward += reward

                if episode_over:
                    if reward > 0: 
                        successes += 1
                        #print ("warm_start simulation episode %s: Success" % (episode))
                    #else: print ("warm_start simulation episode %s: Fail" % (episode))

                    turnsCount = clientClass.send(pickle.dumps(""), "cumulative_turns")
                    turnsCount = int(turnsCount)
                    cumulative_turns+= turnsCount     

            warm_start_run_epochs += 1
            
            
            experience_replay_pool_no = clientClass.send(pickle.dumps(""), "experience_replay_pool")
            experience_replay_pool_no = int(experience_replay_pool_no)

            experience_replay_pool_size = clientClass.send(pickle.dumps(""), "experience_replay_pool_size")
            experience_replay_pool_size = int(experience_replay_pool_size)
            
            if experience_replay_pool_no >= experience_replay_pool_size:
                break
            
        clientClass.send(pickle.dumps(""), "agent_warm_start")
        
        res['success_rate'] = float(successes)/warm_start_run_epochs
        res['ave_reward'] = float(cumulative_reward)/warm_start_run_epochs
        res['ave_turns'] = float(cumulative_turns)/warm_start_run_epochs
        print ("Warm_Start %s epochs, success rate %s, ave reward %s, ave turns %s" % (episode+1, res['success_rate'], res['ave_reward'], res['ave_turns']))
        print ("Current experience replay buffer size %s" % (experience_replay_pool_no))



    def save_time(path, records):
            filename = 'ClientServerTime.json' 
            filepath = os.path.join(path, filename)
            time_str = str(records)
            try:
                json.dump(time_str, open(filepath, "w"))
                print ('Time of the episode saved in %s' % (filepath, ))
            except Exception as e:
                print ('Error: Writing model fails: %s' % (filepath, ))
                print (e)

    def run_episodes(count, status):
        
        successes = 0
        cumulative_reward = 0
        cumulative_turns = 0
        
        if agt == 9 and params['trained_model_path'] == None and warm_start == 1:
            print ('warm_start starting ...')
            warm_start_simulation()
            print ('warm_start finished, start RL training ...')
        startTime = dt.datetime.now()
        for episode in range(count):
            startTimeEpisode = dt.datetime.now()
            print('')
            print ("Episode: %s" % (episode))
            clientClass.send(pickle.dumps(""), "connect")
           #dialog_manager.initialize_episode()
           #print("ONE")
            actionOfUser = dialog_managerClient.initialize_episode()
        #   I need to send to the server that he initialize the dialog_managerServer and send him actionOfUser
        #   Server is now supposed to 1) initialzie_episode
            actionOfUser = pickle.dumps(actionOfUser)
            clientClass.send(actionOfUser, "initialize")
         #   dialog_managerServer.initialize_episode(actionOfUser)
            episode_over = False
            while(not episode_over):
                
        #   I need to send to the server that the does the next turn and returns with the historyDictionary  
                str = ""
                str = pickle.dumps(str)  
                historydictionary = clientClass.send(str , "nextturn1")
               
                #historydictionary = pickle.loads(historydictionary)

                #json.loads(request)
                actionUser2,episode_over, reward = dialog_managerClient.next_turn(historydictionary)

        #   I need to send to the server what the user said and to update the state tracker

                nextturn2Object = {"actionUser2": actionUser2, "episode_over": episode_over, "reward": reward}
               
                
                nextturn2ObjectStr = pickle.dumps(nextturn2Object)
               
                clientClass.send(nextturn2ObjectStr, "nextturn2")

                #dialog_managerServer.next_turnCont(actionUser2, episode_over, reward)

                cumulative_reward += reward
                    
                if episode_over:
                    if reward > 0:
                        print ("Successful Dialog!")
                        successes += 1
                    else: print ("Failed Dialog!")
                    clientClass.send(pickle.dumps(""), "disconnect")
        # I need to send the server to give me the turn count  
                    str = ""
                    str = pickle.dumps(str)   
                    turnsCount = clientClass.send(str, "cumulative_turns")
                    turnsCount = int(turnsCount)
                    cumulative_turns+= turnsCount
                   # cumulative_turns += dialog_managerServer.state_tracker.turn_count
            
           

            if agt == 9 and params['trained_model_path'] == None:

                clientClass.send(pickle.dumps(""), "agent_predict_mode_True")

                simulation_res = simulation_epoch(simulation_epoch_size)
                
                performance_records['success_rate'][episode] = simulation_res['success_rate']
                performance_records['ave_turns'][episode] = simulation_res['ave_turns']
                performance_records['ave_reward'][episode] = simulation_res['ave_reward']
                performance_records['ave_reward'][episode] = simulation_res['ave_reward']
               
                
                if simulation_res['success_rate'] >= best_res['success_rate']:
                    if simulation_res['success_rate'] >= success_rate_threshold: # threshold = 0.30
                        clientClass.send(pickle.dumps(""), "agent_experience_replay_pool_reset")
                        simulation_epoch(simulation_epoch_size)
                    
                if simulation_res['success_rate'] > best_res['success_rate']:

                    best_res['success_rate'] = simulation_res['success_rate']
                    best_res['ave_reward'] = simulation_res['ave_reward']
                    best_res['ave_turns'] = simulation_res['ave_turns']
                    best_res['epoch'] = episode
                    
                clientClass.send(pickle.dumps(""), "agent_clone_dqn")
                clientClass.send(pickle.dumps(batch_size), "agent_train")
                clientClass.send(pickle.dumps(""), "agent_predict_mode_False")
                
                
                print ("Simulation success rate %s, Ave reward %s, Ave turns %s, Best success rate %s" % (performance_records['success_rate'][episode], performance_records['ave_reward'][episode], performance_records['ave_turns'][episode], best_res['success_rate']))
                if episode % save_check_point == 0 and params['trained_model_path'] == None: # save the model every 10 episodes
                   
                   best_res_episode_agt = {'best_res': best_res, 'episode': episode, 'agt':agt}
                   clientClass.send(pickle.dumps(best_res_episode_agt), "save_model")


                   agt_performance_records = {'agt': agt, 'performance_records': performance_records}
                   clientClass.send(pickle.dumps(agt_performance_records), "save_performance_records")

            endTime = dt.datetime.now()#Added to count time


            print ('Time of the episode :',endTime - startTimeEpisode) #Added to count time
            time_taken = endTime - startTimeEpisode
            hours = time_taken.seconds // 3600
            minutes = (time_taken.seconds % 3600) // 60
            seconds = time_taken.seconds % 60
            stringtime = f"{minutes} minutes {seconds} seconds"
            time['time_taken'][episode] = stringtime

            time_taken = endTime - startTime
            hours = time_taken.seconds // 3600
            minutes = (time_taken.seconds % 3600) // 60
            seconds = time_taken.seconds % 60
            stringtime = f"{hours} hours {minutes} minutes {seconds} seconds"
            time['time_taken_so_far'][episode] = stringtime
            time['time_taken_so_far_minutes'][episode] = minutes
            if(params["act_level"]) == 1:
                save_time('./deep_dialog/checkpoints/rl_agent/clientServerAgent/NLP', time)
            else:
                save_time('./deep_dialog/checkpoints/rl_agent/clientServerAgent/noNLP', time)

            print ('Time so far :',endTime - startTime) #Added to count time
            print("Progress: %s / %s, Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (episode+1, count, successes, episode+1, float(cumulative_reward)/(episode+1), float(cumulative_turns)/(episode+1)))
        print("Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (successes, count, float(cumulative_reward)/count, float(cumulative_turns)/count))
        status['successes'] += successes
        status['count'] += count
        
        if agt == 9 and params['trained_model_path'] == None:
        
            agt_division_best_res_count = {'agt': agt, 'division': float(successes)/count, 'best_res': best_res, 'count':count}
            clientClass.send(pickle.dumps(agt_division_best_res_count), "save_model_2")

            agt_performance_records = {'agt': agt, 'performance_records': performance_records}
            clientClass.send(pickle.dumps(agt_performance_records), "save_performance_records")
        
run_episodes(num_episodes, status)
