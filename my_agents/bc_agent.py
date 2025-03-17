#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""
from gc import enable

import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import os
from my_agents.autonomous_agent import AutonomousAgent, control_to_vector, vector_to_control, noop_control
from models.linear_models import Encoder, Decoder, AutoEncoder
from gaze.gaze_utils import apply_gmd_dropout

class BCAgent(AutonomousAgent):

    """
    BC agent to control the ego vehicle using the trained model
    """
    def setup(self, args):
        """
        Setup the agent parameters
        """
        super(BCAgent, self).setup(args)
        with open(args.params_path + '/params.json') as f:
            params = json.load(f)
        
        self.gaze_method = params['gaze_method']
        self.dp_method = params['dp_method']
        self.grayscale = params['grayscale']
        self.stack = params['stack']
        self.embedding_dim = params['embedding_dim']
        self.num_embeddings = params['num_embeddings']
        self.num_hiddens = params['num_hiddens']
        self.num_residual_layers = params['num_residual_layers']
        self.num_residual_hiddens = params['num_residual_hiddens']
        self.z_dim = params['z_dim']
        self.gaze_predictor_path = params['gaze_predictor_path']
        self.models_path = params['models_path']
        self.epochs = params['epochs']
        self.action_dim = params['action_dim']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.raw_files != '':
            self.z_list = []
            self.encoder_input_list = []
        
        if self.gaze_method in ['ViSaRL', 'Mask', 'AGIL'] or self.dp_method in ['GMD', 'IGMD']:
            encoder_gp = Encoder(self.stack * (1 if self.grayscale else 3), self.embedding_dim, self.num_hiddens, self.num_residual_layers, self.num_residual_hiddens,)
            decoder_gp = Decoder(self.stack * (1 if self.grayscale else 3), self.embedding_dim, self.num_hiddens, self.num_residual_layers, self.num_residual_hiddens,)
            self.gaze_predictor = AutoEncoder(encoder_gp, decoder_gp).to(self.device)
            self.gaze_predictor.load_state_dict(torch.load(self.gaze_predictor_path, weights_only=True))
            self.gaze_predictor.eval()
                
        coeff = 2 if self.gaze_method == 'ViSaRL' else 1
        self.encoder = Encoder(coeff * self.stack * (1 if self.grayscale else 3), self.embedding_dim, self.num_hiddens, self.num_residual_layers, self.num_residual_hiddens).to(self.device)
        encoder_output_dim = 20 * 38 * self.embedding_dim
        self.pre_actor = nn.Sequential( nn.Flatten(start_dim=1), nn.Linear(encoder_output_dim, self.z_dim)).to(self.device)
        self.actor = nn.Sequential( nn.Linear(self.z_dim, self.z_dim), nn.ReLU(), nn.Linear(self.z_dim, self.action_dim),).to(self.device)
        self.encoder_agil = None
        if self.gaze_method  == 'AGIL':
            self.encoder_agil = Encoder(self.stack * (1 if self.grayscale else 3), self.embedding_dim, self.num_hiddens, self.num_residual_layers, self.num_residual_hiddens).to(self.device)

        
        self.encoder.load_state_dict(torch.load(self.models_path + "/ep{}_encoder.pth".format(self.epochs), weights_only=True))
        self.pre_actor.load_state_dict(torch.load(self.models_path + "/ep{}_pre_actor.pth".format(self.epochs), weights_only=True))
        self.actor.load_state_dict(torch.load(self.models_path + "/ep{}_actor.pth".format(self.epochs), weights_only=True))
        if self.gaze_method  == 'AGIL':
            self.encoder_agil.load_state_dict(torch.load(self.models_path + "/ep{}_encoder_agil.pth".format(self.epochs), weights_only=True))
                
        
        self.encoder.eval()
        self.pre_actor.eval()
        self.actor.eval()
        if self.gaze_method  == 'AGIL':
            self.encoder_agil.eval()
            
        self.frames_stack = []
                
        
        print("Params are:", params)
        print("                ")
        # exit()
        
    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """

        
        image_center = input_data['Center'][1][:, :, -2::-1]        
        self.frames_to_record.append(image_center)
        self.agent_engaged = True
        
        obs = input_data['Center'][1][:, :, -2::-1]
        if self.obs_res_c == 1:
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)

        if self.obs_res_w != self.render_res_w or self.obs_res_h != self.render_res_h:
            obs = cv2.resize(obs, (self.obs_res_w, self.obs_res_h))
        
        if len(self.frames_stack) == 0:
            for _ in range(self.stack):
                self.frames_stack.append(obs.copy())
        else:
            self.frames_stack.append(obs)
            self.frames_stack.pop(0)
        
        stacked_obs = np.stack(self.frames_stack, axis=0)
        stacked_obs = torch.from_numpy(stacked_obs.copy()).permute(0, 3, 1, 2)
        if self.grayscale:
            stacked_obs = stacked_obs.float().mean(1, keepdim=True).to(torch.uint8)
        stacked_obs = (stacked_obs.float() / 255.0).to(self.device)
        stacked_obs = stacked_obs.reshape(-1, stacked_obs.shape[-2], stacked_obs.shape[-1])
        stacked_obs = stacked_obs.unsqueeze(0)
        with torch.no_grad():        
            if self.gaze_method in ['ViSaRL', 'Mask', 'AGIL'] or self.dp_method in ['GMD', 'IGMD']: # models that need gaze prediction
                g = self.gaze_predictor(stacked_obs)
                g[g < 0] = 0
                g[g > 1] = 1
            
            
            if self.gaze_method == 'ViSaRL':
                stacked_obs = torch.cat([stacked_obs, g], dim=1)
            elif self.gaze_method == 'Mask':
                stacked_obs = stacked_obs * g
            
            dropout_mask = None
            if self.dp_method == 'IGMD':
                dropout_mask = g[:,-1:]

            if self.raw_files != '':
                self.encoder_input_list.append(stacked_obs.cpu().numpy())
            z = self.encoder(stacked_obs, dropout_mask=dropout_mask)

            if self.raw_files != '':
                self.z_list.append(z.cpu().numpy())

            if self.gaze_method == 'AGIL':
                z = (z + self.encoder_agil(stacked_obs * g)) / 2
            
            if self.dp_method == 'GMD':
                z = apply_gmd_dropout(z, g[:,-1:], test_mode=True)

            z = self.pre_actor(z)
            action = self.actor(z)
            
        v = action.cpu()[0].numpy()
        # print('v:', v)
        control = vector_to_control(v)
        
        self.steps += 1
        if self.steps < 10:
            return noop_control()
        
        elif self.steps > self.fps * 100:
            raise Exception("BCAgent failed to finish the route")

        return control




    def destroy(self):
        """
        Cleanup
        """
        
        if self.raw_files != '':
            np.save(self.raw_files + '/z_list.npy', np.array(self.z_list))
            self.z_list = []

            np.save(self.raw_files + '/encoder_input_list.npy', np.array(self.encoder_input_list))
            self.encoder_input_list = []
                
        torch.cuda.empty_cache()
        super(BCAgent, self).destroy()
