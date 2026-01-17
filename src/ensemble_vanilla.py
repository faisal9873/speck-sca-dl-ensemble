
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import pandas as pd
from tensorflow.keras import backend as backend
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

sys.path.append('..')  # Add parent directory to the Python path

import sca_metrics
from sca_metrics import SCAMetrics

import loaddataset
from loaddataset import LoadDatasets

import datasets
from datasets import SCADatasets

import neural_networks
from neural_networks import NeuralNetwork
sys.path.pop()


import numpy as np
import random


class EnsembleAES:

    def __init__(self):
        self.number_of_models = 50
        self.number_of_best_models = [1, 5, 10, 20]#[10, 10, 10, 10]
        self.ge_all_validation = []
        self.ge_all_attack = []
        self.sr_all_validation = []
        self.sr_all_attack = []
        self.k_ps_all = []
        self.model_all =[] #added, expected to store all models
        self.model_all_ranked = []
        self.ge_ensemble = None
        self.ge_ensemble_best_models = None
        self.ge_ensemble_best_models0 = None #added
        self.ge_ensemble_best_models1 = None #added
        self.ge_ensemble_best_models2 = None #added
        self.ge_ensemble_best_models3 = None #added
        self.ge_best_model_validation = None
        self.ge_best_model_attack = None
        self.sr_ensemble = None
        self.sr_ensemble_best_models = None
        self.sr_ensemble_best_models0 = None #added
        self.sr_ensemble_best_models1 = None #added
        self.sr_ensemble_best_models2 = None #added
        self.sr_ensemble_best_models3 = None #added
        self.sr_best_model_validation = None
        self.sr_best_model_attack = None
        self.target_dataset = None
        self.l_model = None
        self.target_byte = None
        self.classes = None
        self.epochs = None
        self.mini_batch = None

    def set_dataset(self, target): #function for setting up dataset - ASCAD | ches_ctf
        self.target_dataset = target

    def set_leakage_model(self, leakage_model): #function for setting up leakage model
        self.l_model = leakage_model
        if leakage_model == "HW":
            self.classes = 9
        else:
            self.classes = 256

    def set_target_byte(self, target_byte): #target byte to attack: most of the time, this is 3
        self.target_byte = target_byte

    def set_epochs(self, epochs): #we set no of epochs for our training model
        self.epochs = epochs

    def set_mini_batch(self, mini_batch): #not sure of mini_batch. I guess it means batch
        self.mini_batch = mini_batch

    def __add_if_one(self, value): #don't know when this will be useful
        return 1 if value == 1 else 0

    
    def get_best_models(self, n_models, result_models_validation, n_traces):
        result_number_of_traces_val = []  # Best models ranked by guessing entropy traces
    
        for model_index in range(n_models):
            added = False  # Flag to track if the model has been added to the list
    
            if result_models_validation[model_index][n_traces - 1] == 1:  # Checking for best model
                for index in range(n_traces - 1, -1, -1):  # Decrement and move up traces
                    if result_models_validation[model_index][index] != 1:
                        result_number_of_traces_val.append(
                            [result_models_validation[model_index][n_traces - 1], index + 1, model_index]
                        )
                        added = True
                        break
    
            if not added:  # If no break or `if` was satisfied, use the fallback case
                result_number_of_traces_val.append(
                    [result_models_validation[model_index][n_traces - 1], n_traces, model_index]
                )
    
        # Sort models
        sorted_models = sorted(result_number_of_traces_val, key=lambda l: l[:])
        print(f'Length of sorted models: {len(sorted_models)}')
    
        # Extract best models
        list_of_best_models = []
        list_of_best_models1 = []
        print(f'Length of n_models: {n_models}')
    
        for model_index in range(n_models):
            list_of_best_models.append(sorted_models[model_index][2])
            list_of_best_models1.append(self.model_all[sorted_models[model_index][2]])
    
        return list_of_best_models, list_of_best_models1

    def run_mlp(self, X_profiling, Y_profiling, X_validation, Y_validation, X_attack, Y_attack, plt_validation, plt_attack, params,
                step, fraction):
        mini_batch = random.randrange(500, 1000, 100)
        learning_rate = random.uniform(0.0001, 0.001)
        activation = ['relu', 'tanh', 'elu', 'selu'][random.randint(0, 3)]
        layers = random.randrange(2, 8, 1)
        neurons = random.randrange(500, 800, 100)

        model = NeuralNetwork().mlp_random(self.classes, params["number_of_samples"], activation, neurons, layers, learning_rate)
        model.fit(
            x=X_profiling,
            y=Y_profiling,
            batch_size=self.mini_batch,
            verbose=1,
            epochs=self.epochs,
            shuffle=True,
            validation_data=(X_validation, Y_validation),
            callbacks=[])
        model_trained = model

        ge_validation, sr_validation, kp_krs = SCAMetrics().ge_and_sr(100, model, params, self.l_model, self.target_byte,
                                                                      X_validation, plt_validation, step, fraction)
        ge_attack, sr_attack, _ = SCAMetrics().ge_and_sr(100, model, params, self.l_model, self.target_byte, X_attack, plt_attack, step,
                                                         fraction)

        backend.clear_session()

        return ge_validation, ge_attack, sr_validation, sr_attack, kp_krs, model_trained #added model

    def run_cnn(self, X_profiling, Y_profiling, X_validation, Y_validation, X_attack, Y_attack, plt_validation, plt_attack, params,
                step, fraction):
        X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
        X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))
        X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

        mini_batch = random.randrange(500, 1000, 100)
        learning_rate = random.uniform(0.0001, 0.001)
        activation = ['relu', 'tanh', 'elu', 'selu'][random.randint(0, 3)]
        dense_layers = random.randrange(2, 8, 1)
        neurons = random.randrange(500, 800, 100)
        conv_layers = random.randrange(1, 2, 1)
        filters = random.randrange(8, 32, 4)
        kernel_size = random.randrange(10, 20, 2)
        stride = random.randrange(5, 10, 5)

        model = NeuralNetwork().cnn_random(self.classes, params["number_of_samples"], activation, neurons, conv_layers, filters,
                                           kernel_size, stride, dense_layers, learning_rate)
        model.fit(
            x=X_profiling,
            y=Y_profiling,
            batch_size=self.mini_batch,
            verbose=1,
            epochs=self.epochs,
            shuffle=True,
            validation_data=(X_validation, Y_validation),
            callbacks=[])
        model_trained = model
                                                                      #100 is for number of runs
        ge_validation, sr_validation, kp_krs = SCAMetrics().ge_and_sr(100, model, params, self.l_model, self.target_byte,
                                                                      X_validation, plt_validation,
                                                                      step, fraction)
        ge_attack, sr_attack, _ = SCAMetrics().ge_and_sr(100, model, params, self.l_model, self.target_byte, X_attack, plt_attack, step,
                                                         fraction)

        backend.clear_session()

        return ge_validation, ge_attack, sr_validation, sr_attack, kp_krs, model_trained #added model

    def compute_ensembles(self, kr_nt, correct_key):
       #nm = number_of_models
      list_of_best_models, list_of_best_models1 = self.get_best_models(self.number_of_models, self.ge_all_validation, kr_nt) #i added #added list_of_best_models1
      
      self.model_all_ranked = list_of_best_models1 #added
      self.ge_best_model_validation = self.ge_all_validation[list_of_best_models[0]]
      self.ge_best_model_attack = self.ge_all_attack[list_of_best_models[0]]
      self.sr_best_model_validation = self.sr_all_validation[list_of_best_models[0]]
      self.sr_best_model_attack = self.sr_all_attack[list_of_best_models[0]]

      for i in range(4): #added

          kr_ensemble = np.zeros(kr_nt)
          krs_ensemble = np.zeros((100, kr_nt))
          kr_ensemble_best_models = np.zeros(kr_nt)
          krs_ensemble_best_models = np.zeros((100, kr_nt))

          for run in range(100):

              key_p_ensemble = np.zeros(256)
              key_p_ensemble_best_models = np.zeros(256)

              for index in range(kr_nt):
                  for model_index in range(self.number_of_models):
                      key_p_ensemble += np.log(self.k_ps_all[list_of_best_models[model_index]][run][index] + 1e-36)
                  for model_index in range(self.number_of_best_models[i]): #i added
                      key_p_ensemble_best_models += np.log(self.k_ps_all[list_of_best_models[model_index]][run][index] + 1e-36)

                  key_p_ensemble_sorted = np.argsort(key_p_ensemble)[::-1]
                  key_p_ensemble_best_models_sorted = np.argsort(key_p_ensemble_best_models)[::-1]

                  kr_position = list(key_p_ensemble_sorted).index(correct_key) + 1
                  kr_ensemble[index] += kr_position
                  krs_ensemble[run][index] = kr_position

                  kr_position = list(key_p_ensemble_best_models_sorted).index(correct_key) + 1
                  kr_ensemble_best_models[index] += kr_position
                  krs_ensemble_best_models[run][index] = kr_position

              print("Run {} - GE {} models: {} | GE {} models: {} | ".format(run, self.number_of_models,
                                                                            int(kr_ensemble[kr_nt - 1] / (run + 1)),
                                                                            self.number_of_best_models[i], #i added
                                                                            int(kr_ensemble_best_models[kr_nt - 1] / (run + 1))))

          ge_ensemble = kr_ensemble / 100
          ge_ensemble_best_models = kr_ensemble_best_models / 100

          sr_ensemble = np.zeros(kr_nt)
          sr_ensemble_best_models = np.zeros(kr_nt)

          for index in range(kr_nt):
              for run in range(100):
                  sr_ensemble[index] += self.__add_if_one(krs_ensemble[run][index])
                  sr_ensemble_best_models[index] += self.__add_if_one(krs_ensemble_best_models[run][index])
          if (i==0):
            ge_ensemble_best_models0 = ge_ensemble_best_models
            sr_ensemble_best_models0 = sr_ensemble_best_models
          if (i==1):
            ge_ensemble_best_models1 = ge_ensemble_best_models
            sr_ensemble_best_models1 = sr_ensemble_best_models
          if(i==2):
            ge_ensemble_best_models2 = ge_ensemble_best_models
            sr_ensemble_best_models2 = sr_ensemble_best_models
          if(i==3):
            ge_ensemble_best_models3 = ge_ensemble_best_models
            sr_ensemble_best_models3 = sr_ensemble_best_models
            ge_ensemble_best_models = ge_ensemble_best_models0
            sr_ensemble_best_models = sr_ensemble_best_models0

        #return ge_ensemble, ge_ensemble_best_models, sr_ensemble/100, sr_ensemble_best_models/100
          #next line added
      return ge_ensemble, ge_ensemble_best_models, ge_ensemble_best_models1, ge_ensemble_best_models2, ge_ensemble_best_models3, sr_ensemble/100, sr_ensemble_best_models/100, sr_ensemble_best_models1/100,sr_ensemble_best_models2/100,sr_ensemble_best_models3/100

    def create_z_score_norm(self, dataset):
        z_score_mean = np.mean(dataset, axis=0)
        z_score_std = np.std(dataset, axis=0)
        return z_score_mean, z_score_std

    def apply_z_score_norm(self, dataset, z_score_mean, z_score_std):
        for index in range(len(dataset)):
            dataset[index] = (dataset[index] - z_score_mean) / z_score_std

    def run_ensemble(self, number_of_models, number_of_best_models):

        self.number_of_models = number_of_models
        self.number_of_best_models = number_of_best_models

        target_params = SCADatasets().get_trace_set(self.target_dataset)

        root_folder = ''

        (X_profiling, Y_profiling), (X_validation, Y_validation), (X_attack, Y_attack), (
            _, plt_validation, plt_attack) = LoadDatasets().load_dataset(
            root_folder + target_params["file"], target_params["n_profiling"], target_params["n_attack"], self.target_byte, self.l_model)


        # normalize with z-score
        z_score_mean, z_score_std = self.create_z_score_norm(X_profiling)
        self.apply_z_score_norm(X_profiling, z_score_mean, z_score_std) #a added
        self.apply_z_score_norm(X_validation, z_score_mean, z_score_std)
        self.apply_z_score_norm(X_attack, z_score_mean, z_score_std)

        # convert labels to categorical labels
       
        Y_profiling = to_categorical(Y_profiling, num_classes=self.classes) #a added
        Y_validation = to_categorical(Y_validation, num_classes=self.classes)
        Y_attack = to_categorical(Y_attack, num_classes=self.classes)

        X_profiling = X_profiling.astype('float32')
        X_validation = X_validation.astype('float32')
        X_attack = X_attack.astype('float32')

        kr_step = 5  # key rank processed for each kr_step traces #changed this from 5 to 10
        kr_fraction = 1  # validation or attack sets are divided by kr_fraction before computing key rank

        self.ge_all_validation = []
        self.sr_all_validation = []
        self.ge_all_attack = []
        self.k_ps_all = []

        kr_nt = int(len(X_validation) / (kr_step * kr_fraction))

        # # train random MLP
        # for model_index in range(self.number_of_models):
        #     ge_validation, ge_attack, sr_validation, sr_attack, kp_krs, model = self.run_mlp(X_profiling, Y_profiling,  #added model
        #                                                                               X_validation, Y_validation,
        #                                                                               X_attack, Y_attack,
        #                                                                               plt_validation, plt_attack,
        #                                                                               target_params, kr_step, kr_fraction)
        #     self.ge_all_validation.append(ge_validation)
        #     self.ge_all_attack.append(ge_attack)
        #     self.sr_all_validation.append(sr_validation)
        #     self.sr_all_attack.append(sr_attack)
        #     self.k_ps_all.append(kp_krs)
        #     self.model_all.append(model)

        # train random CNN
        for model_index in range(self.number_of_models):
            ge_validation, ge_attack, sr_validation, sr_attack, kp_krs, model = self.run_cnn(X_profiling, Y_profiling, #added model
                                                                                      X_validation, Y_validation,
                                                                                      X_attack, Y_attack,
                                                                                      plt_validation, plt_attack,
                                                                                      target_params, kr_step, kr_fraction)
            self.ge_all_validation.append(ge_validation)
            self.ge_all_attack.append(ge_attack)
            self.sr_all_validation.append(sr_validation)
            self.sr_all_attack.append(sr_attack)
            self.k_ps_all.append(kp_krs)                  #kr_nt = int(len(X_validation) / (kr_step * kr_fraction))
            self.model_all.append(model)

        #ge_ensemble, ge_ensemble_best_models, sr_ensemble, sr_ensemble_best_models = self.compute_ensembles(kr_nt,
         #                                                                                                   target_params["good_key"])
        ge_ensemble, ge_ensemble_best_models, ge_ensemble_best_models1, ge_ensemble_best_models2, ge_ensemble_best_models3, sr_ensemble, sr_ensemble_best_models, sr_ensemble_best_models1, sr_ensemble_best_models2,sr_ensemble_best_models3 = self.compute_ensembles(kr_nt,
                                                                                                             target_params["good_key"])


        self.ge_ensemble = ge_ensemble
        self.ge_ensemble_best_models = ge_ensemble_best_models
        self.ge_ensemble_best_models1 = ge_ensemble_best_models1 #added
        self.ge_ensemble_best_models2 = ge_ensemble_best_models2 #added
        self.ge_ensemble_best_models3 = ge_ensemble_best_models3 #added

        self.sr_ensemble = sr_ensemble
        self.sr_ensemble_best_models = sr_ensemble_best_models
        self.sr_ensemble_best_models1 = sr_ensemble_best_models1 #added
        self.sr_ensemble_best_models2 = sr_ensemble_best_models2 #added
        self.sr_ensemble_best_models3 = sr_ensemble_best_models3 #added




    def get_ge_ensemble(self):
        return self.ge_ensemble

    def get_ge_ensemble_best_models(self):
        return self.ge_ensemble_best_models

    def get_ge_ensemble_best_models1(self): #added
        return self.ge_ensemble_best_models1

    def get_ge_ensemble_best_models2(self): #added
        return self.ge_ensemble_best_models2

    def get_ge_ensemble_best_models3(self): #added
        return self.ge_ensemble_best_models3

    def get_ge_best_model_validation(self):
        return self.ge_best_model_validation

    def get_ge_best_model_attack(self):
        return self.ge_best_model_attack

    def get_sr_ensemble(self):
        return self.sr_ensemble

    def get_sr_ensemble_best_models1(self): #added
        return self.sr_ensemble_best_models1

    def get_sr_ensemble_best_models2(self): #added
        return self.sr_ensemble_best_models2

    def get_sr_ensemble_best_models3(self): #added
        return self.sr_ensemble_best_models3


    def get_sr_ensemble_best_models(self):
        return self.sr_ensemble_best_models

    def get_sr_best_model_validation(self):
        return self.sr_best_model_validation

    def get_sr_best_model_attack(self):
        return self.sr_best_model_attack
# added
dataseta = "SPECK_variable_key" #"SPECK_fixed_key"  #"SPECK_fixed_key" # 
target_byte = 7 #2 for ascad, #0 for ches

for k in range(1):
    ensemble_aes = EnsembleAES()
    ensemble_aes.set_dataset(dataseta)  # "SPECK_fixed_key", "SPECK_variable_key", "ascad_fixed_key", "ascad_random_key" or "ches_ctf"
    ensemble_aes.set_leakage_model("HW") #HW
    ensemble_aes.set_target_byte(target_byte)
    ensemble_aes.set_mini_batch(400)
    ensemble_aes.set_epochs(10)
    #ensemble_aes.run_ensemble(number_of_models=5, number_of_best_models=[1,2,3,4])
    ensemble_aes.run_ensemble(number_of_models=30, number_of_best_models=[1,5,10,20])
    # for model in ensemble_aes.model_all_ranked:  #added to view the ranked models
    #     model.summary()

  
    plt.clf()
    plt.rc('figure', figsize=(15, 8))
    
    # Define line styles for different plots
    line_styles = ['-', '--', '-.', ':', (0, (1, 10))]
    

    # Plotting GE and SR with different line styles
    plt.plot(range(ensemble_aes.get_ge_ensemble().shape[0]), ensemble_aes.get_ge_ensemble_best_models(), label="Basic E1", linestyle=line_styles[0])
    plt.plot(range(ensemble_aes.get_ge_ensemble().shape[0]), ensemble_aes.get_ge_ensemble_best_models1(), label="Basic E5", linestyle=line_styles[1])
    plt.plot(range(ensemble_aes.get_ge_ensemble().shape[0]), ensemble_aes.get_ge_ensemble_best_models2(), label="Basic E10", linestyle=line_styles[2])
    plt.plot(range(ensemble_aes.get_ge_ensemble().shape[0]), ensemble_aes.get_ge_ensemble_best_models3(), label="Basic E20", linestyle=line_styles[3])
    plt.plot(range(ensemble_aes.get_ge_ensemble().shape[0]), ensemble_aes.get_ge_ensemble(), label="Basic E50", linestyle=line_styles[4])


    df = pd.DataFrame (ensemble_aes.get_ge_ensemble()) #added to save data
    df1 = pd.DataFrame (ensemble_aes.get_ge_ensemble_best_models())
    df2 = pd.DataFrame (ensemble_aes.get_ge_ensemble_best_models1())
    df3 = pd.DataFrame (ensemble_aes.get_ge_ensemble_best_models2())
    df4 = pd.DataFrame (ensemble_aes.get_ge_ensemble_best_models3())
    ## save to xlsx file
    
    filepath = 'E50_cnn_basic'+'_'+str(k)+'_'+'.xlsx'
    filepath1 = 'E1_cnn_basic'+'_'+str(k)+'_'+'.xlsx'
    filepath2 = 'E5_cnn_basic'+'_'+str(k)+'_'+'.xlsx'
    filepath3 = 'E10_cnn_basic'+'_'+str(k)+'_'+'.xlsx'
    filepath4 = 'E20_cnn_basic'+'_'+str(k)+'_'+'.xlsx'
    
    df.to_excel(filepath, index=False)
    df1.to_excel(filepath1, index=False)
    df2.to_excel(filepath2, index=False)
    df3.to_excel(filepath3, index=False)
    df4.to_excel(filepath4, index=False)
    
    plt.xlabel("Traces")
    plt.ylabel("Guessing Entropy")
    plt.legend()
    plt.grid()
 
    plt.savefig(dataseta +'_' +'cnn'+'_'+'basic'+'_'+str(k)+'_'+'.png')

