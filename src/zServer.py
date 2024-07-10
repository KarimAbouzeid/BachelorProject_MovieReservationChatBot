import socket
import threading
import sys
import argparse, json, copy, os
import pickle
import ast 
from deep_dialog.dialog_system import DialogManager, text_to_dict, DialogManagerServer
from deep_dialog.agents import AgentCmd, InformAgent, RequestAllAgent, RandomAgent, EchoAgent, RequestBasicsAgent, AgentDQN
from deep_dialog.usersims import RuleSimulator

from deep_dialog import dialog_config
from deep_dialog.dialog_config import *

from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg



HEADER = 64
PORT = 5555
DISCONNECT_MESSAGE = "!DISCONNECT"
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)




class zServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.dialog_managerServer = ''
        self.agent=''
        self.write_model_dir = ''
        self.best_model = ''
        self.params = ''

    def start(self):
        # create a socket object
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # bind the socket to a specific IP address and port number
        server_socket.bind((self.host, self.port))
        # listen for incoming client connections
        server_socket.listen(1)

        print(f"Server listening on {self.host}:{self.port}")

        while True:
            # accept incoming client connection
            client_socket, address = server_socket.accept()
        ###    #print(f"Connection from {address}")
            # handle the client request in a separate thread
        ###    # client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
        ###   # client_thread.start()
            self.handle_client(client_socket)

    



            
       
    def initialize_episode (self, actionOfUser):
        self.dialog_managerServer.initialize_episode(actionOfUser)
    
    def nextturn1 (self):
         history = self.dialog_managerServer.next_turn()
         return history
    
    def nextturn2 (self, nextturn2Object):
         actionUser2 = nextturn2Object["actionUser2"]
         episode_over = nextturn2Object["episode_over"]
         reward = nextturn2Object["reward"]
         self.dialog_managerServer.next_turnCont(actionUser2, episode_over, reward)

    def updateTurn (self):
         return self.dialog_managerServer.state_tracker.turn_count
         

    def bestModelUpdate (self):
         self.best_model['model'] = copy.deepcopy(self.agent)



    def save_model(self, path, agt, success_rate, agent, best_epoch, cur_epoch):
        filename = 'agt_%s_%s_%s_%.5f.p' % (agt, best_epoch, cur_epoch, success_rate)
        filepath = os.path.join(path, filename)
        checkpoint = {}
        if agt == 9: checkpoint['model'] = copy.deepcopy(agent.dqn.model)
        checkpoint['params'] = self.params
        try:
            pickle.dump(checkpoint, open(filepath, "wb"))
            print ('saved model in %s' % (filepath, ))
        except Exception as e:
            print ('Error: Writing model fails: %s' % (filepath, ))
            print (e)


    """ save performance numbers """
    def save_performance_records(self, path, agt, records):
        filename = 'agt_%s_performance_records.json' % (agt)
        filepath = os.path.join(path, filename)
        try:
            json.dump(records, open(filepath, "w"))
            print ('saved model in %s' % (filepath, ))
        except Exception as e:
            print ('Error: Writing model fails: %s' % (filepath, ))
            print (e)
    

    def handle_chatbot (self, params):
       
       self.params = params
       max_turn = params['max_turn']
       num_episodes = params['episodes']

       agt = params['agt']

       dict_path = params['dict_path']
       goal_file_path = params['goal_file_path']


       movie_kb_path = params['movie_kb_path']
       with open(movie_kb_path, "rb") as f:
             movie_kb = pickle.load(f, encoding="latin1")

       act_set = text_to_dict(params['act_set'])
       slot_set = text_to_dict(params['slot_set'])
       ################################################################################
       # a movie dictionary for user simulator - slot:possible values
       ################################################################################
       movie_dictionary = pickle.load(open(dict_path, 'rb'))

       dialog_config.run_mode = params['run_mode']
       dialog_config.auto_suggest = params['auto_suggest']

       ################################################################################
       #   Parameters for Agents
       ################################################################################
       agent_params = {}
       agent_params['max_turn'] = max_turn
       agent_params['epsilon'] = params['epsilon']
       agent_params['agent_run_mode'] = params['run_mode']
       agent_params['agent_act_level'] = params['act_level']

       agent_params['experience_replay_pool_size'] = params['experience_replay_pool_size']
       agent_params['dqn_hidden_size'] = params['dqn_hidden_size']
       agent_params['batch_size'] = params['batch_size']
       agent_params['gamma'] = params['gamma']
       agent_params['predict_mode'] = params['predict_mode']
       agent_params['trained_model_path'] = params['trained_model_path']
       agent_params['warm_start'] = params['warm_start']
       agent_params['cmd_input_mode'] = params['cmd_input_mode']

       self.write_model_dir = params['write_model_dir']

       if agt == 0:
            self.agent = AgentCmd(movie_kb, act_set, slot_set, agent_params)
       elif agt == 1:
            self.agent = InformAgent(movie_kb, act_set, slot_set, agent_params)
       elif agt == 2:
            self.agent = RequestAllAgent(movie_kb, act_set, slot_set, agent_params)
       elif agt == 3:
            self.agent = RandomAgent(movie_kb, act_set, slot_set, agent_params)
       elif agt == 4:
            self.agent = EchoAgent(movie_kb, act_set, slot_set, agent_params)
       elif agt == 5:
            self.agent = RequestBasicsAgent(movie_kb, act_set, slot_set, agent_params)
       elif agt == 9:
            self.agent = AgentDQN(movie_kb, act_set, slot_set, agent_params)
            
        ################################################################################
        #    Add your agent here
        ################################################################################
       else:
            pass

      ################################################################################
      # load trained NLG model
      ################################################################################
       nlg_model_path = params['nlg_model_path']
       diaact_nl_pairs = params['diaact_nl_pairs']
       nlg_model = nlg()
       nlg_model.load_nlg_model(nlg_model_path)
       nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs)
       self.agent.set_nlg_model(nlg_model)

       ################################################################################
        # load trained NLU model
        ################################################################################
       nlu_model_path = params['nlu_model_path']
       nlu_model = nlu()
       nlu_model.load_nlu_model(nlu_model_path)
       self.agent.set_nlu_model(nlu_model)
       self.dialog_managerServer = DialogManagerServer(self.agent, act_set, slot_set, movie_kb)

       save_check_point = params['save_check_point']
      
       """Best Model and Performance Records """
       self.best_model = {}
       self.best_model['model'] = copy.deepcopy(self.agent)
     




    def handle_client(self,client_socket):

        # receive data from the client
        
        request = client_socket.recv(2048)

        # Extract the message header and the data

        
        
        header = request[:6].decode()
   
        request = request[6:]
        
        

        ## Handle ChatBot
        if header == 'params':
            # convert request to JSON
            request = request.decode()
            print(request)
            request = json.loads(request)
            self.handle_chatbot(request)

        ## Initialize Episode
        elif header == 'initia':
             # convert request to JSON
            request = pickle.loads(request)
            self.initialize_episode(request)
            
            

        elif header == 'nxtrn1':
            response = self.nextturn1()
            response = pickle.dumps(response)
            client_socket.sendall(response)

        elif header == 'nxtrn2':
            # convert request to JSON
            request = pickle.loads(request)
            self.nextturn2(request)

        elif header == 'cumtrn':
            client_socket.sendall(str(self.updateTurn()).encode())

        elif header == 'connec':
            print()
            episode = pickle.loads(request)
            print ("Client connected Episode: %s" % (episode))
        elif header == 'discon':
            print("Client disconnected")

        elif header == 'exprpo':
            client_socket.sendall(str(len(self.agent.experience_replay_pool)).encode())
        
        elif header == 'exprps':
            client_socket.sendall(str(self.agent.experience_replay_pool_size).encode())

        elif header == 'agwrst':
            self.agent.warm_start = 2
        
        elif header == 'agprmt':
            self.agent.predict_mode = True

        
        
        elif header == 'agrpre':
            self.agent.experience_replay_pool = [] 

        elif header == 'agprmf':
            self.agent.predict_mode = False

        elif header == 'agcldq':
            self.agent.clone_dqn = copy.deepcopy(self.agent.dqn)

        elif header == 'agtrbs':
            batch_size = pickle.loads(request)
            self.agent.train(batch_size, 1)
        
        elif header == 'agcopy':
           
            agent = pickle.dumps(self.agent)
            size = sys.getsizeof(agent)
            print(f"Size of my_list: {size} bytes")
            client_socket.sendall(agent)

        elif header == 'savmod':
            request = pickle.loads(request)
            best_res = request['best_res']
            episode = request['episode']
            agt = request['agt']
            self.save_model(self.write_model_dir, agt, best_res['success_rate'], self.best_model['model'], best_res['epoch'], episode)


        elif header == 'savper':
            request = pickle.loads(request)
            performance_records = request['performance_records']
            agt = request['agt']
            self.save_performance_records(self.write_model_dir, agt, performance_records)

        elif header == 'savmd2':
            request = pickle.loads(request)
            best_res = request['best_res']
            count = request['count']
            agt = request['agt']
            divison = request['division']
            self.save_model(self.write_model_dir, agt, divison, self.best_model['model'], best_res['epoch'], count)



            


        # # extract parameter from JSON
        # param_value = request.get('param')

        # # process the request and send a response
        # response = f"Hello, client! You sent param = {param_value}"
        # client_socket.send(response.encode())
        # close the client socket
        client_socket.close()

if __name__ == '__main__':
    
    server = zServer(SERVER, PORT)
    server.start()
