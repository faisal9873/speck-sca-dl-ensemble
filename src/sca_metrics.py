
import numpy as np
import random
from sklearn.utils import shuffle


class SCAMetrics:       #s box
    
    def speck_labelize_ge_sr(self, trace_data, byte, round_key, leakage_model):
    # Extract plaintext and key data for the selected byte position
        pt_ct = [int(''.join(f'{int(byte):02X}' for byte in row[0:4]), 16) for row in trace_data]
        key_byte = int(''.join(f'{int(byte):02X}' for byte in round_key), 16)
        key_byte = np.full(len(pt_ct), key_byte)
        key_byte = np.asarray(key_byte[:])

    
        # SPECK uses a 32-bit block, so split plaintext into two 16-bit words
        intermediate_values = []
        for pt, k in zip(pt_ct, key_byte):
            
            # Split 32-bit plaintext block into two 16-bit words
            y = pt & 0xFFFF        # Lower 16 bits
            x = (pt >> 16) & 0xFFFF  # Upper 16 bits
            k = k & 0xFFFF
    
            # Perform one round of SPECK encryption
            x = ((x << 7) | (x >> 9)) & 0xFFFF  # Rotate x left by 7 bits
            x = (x + y) & 0xFFFF                # Addition mod 2^16
            x ^= k                              # XOR with the round key
            y = ((y >> 2) | (y << 14)) & 0xFFFF # Rotate y right by 2 bits
            y ^= x                              # XOR with x
    
            # Combine y and x into a single 32-bit value for the intermediate state
            intermediate_value = (x << 16) | y
            
            # Extract only the lower or upper byte (last 8 bits)
            lower_byte = intermediate_value & 0xFF  # This isolates 0x88 if intermediate_value is 0x7788
            # upper_byte = (intermediate_value >> 8) & 0xFF  # This isolates 0x77 if intermediate_value is 0x7788
            intermediate_values.append(lower_byte)
            # intermediate_values.append(upper_byte)
            

        # Convert intermediate values to Hamming Weight if using HW leakage model
        if leakage_model == "HW":
            hw_values = [bin(iv).count("1") for iv in intermediate_values]
            return hw_values
        else:
            return intermediate_values

    def ge_and_sr(self, runs, model, param, leakage_model, byte, x_test, test_trace_data, step, fraction):
        nt = len(x_test) #number of test data
        nt_kr = int(nt / fraction) #fraction of nt that we want to use for key rank plot
        nt_interval = int(nt / (step * fraction)) #interval, determined by the number of steps given 
        key_ranking_sum = np.zeros(nt_interval)   #observations per step are all initialized to 0
        success_rate_sum = np.zeros(nt_interval)  #observations per step are all initialized to 0
        key_probabilities_key_ranks = np.zeros((runs, nt, 256)) #key rank array is created based on number of runs defined

        # ---------------------------------------------------------------------------------------------------------#
        # compute labels for all key hypothesis
        # ---------------------------------------------------------------------------------------------------------#
        labels_key_hypothesis = np.zeros((256, nt)) #empty array is created and filled with zeros
        for key_byte_hypothesis in range(0, 256): #loop is running 256 times => number of classes
            key_h = bytearray.fromhex(param["key"]) #key is converted to an array of bytes. Param holds the key value
            key_h[byte] = key_byte_hypothesis #select the targetted byte (i.e. 3) from the key (16 bytes)
            # print(f' key_byte_hypothesis: {key_byte_hypothesis}')
            labels_key_hypothesis[key_byte_hypothesis][:] = self.speck_labelize_ge_sr(test_trace_data, byte, key_h, leakage_model)

        # ---------------------------------------------------------------------------------------------------------#
        # predict output probabilities for shuffled test or validation set
        # ---------------------------------------------------------------------------------------------------------#
        output_probabilities = model.predict(x_test) #predict the label for all x_test

        probabilities_kg_all_traces = np.zeros((nt, 256)) #new array created to store x_test. remember nt = len(x_test) 
        for index in range(nt):
            probabilities_kg_all_traces[index] = output_probabilities[index][  #labeling is done trace by trace
                np.asarray([int(leakage[index]) for leakage in labels_key_hypothesis[:]]) #the other classes are predicted with 0
            ]

        for run in range(runs):

            probabilities_kg_all_traces_shuffled = shuffle(probabilities_kg_all_traces, random_state=random.randint(0, 100000))
            key_probabilities = np.zeros(256)
            kr_count = 0
            for index in range(nt_kr): #using the fraction of traces from nt                #1e-36 avoid taking log 0
                key_probabilities += np.log(probabilities_kg_all_traces_shuffled[index] + 1e-36)#log is taken for each trace and then summed
                key_probabilities_key_ranks[run][index] = probabilities_kg_all_traces_shuffled[index]#result arranged as per #run
                key_probabilities_sorted = np.argsort(key_probabilities)[::-1] ##sorting of probabilities, smallest to highest but indices are returned
                if (index + 1) % step == 0:  #loop is entered after every step (of x-axis if it were a plot)
                    key_ranking_good_key = list(key_probabilities_sorted).index(param["good_key"]) + 1 #We take out index of the good key from sorted prob & we add 1 to it
                    key_ranking_sum[kr_count] += key_ranking_good_key  #key rank for the good key is saved here every step. + helps to add up after each run
                    if key_ranking_good_key == 1: #success_rate_sum[kr_count]=1 only when the key has a good rank
                        success_rate_sum[kr_count] += 1          #it is 0 when rank is not 1
                    kr_count += 1
            print(                                                                      #GE is calculated after each run cumulatively
                "KR: {} | GE for correct key ({}): {})".format(run, param["good_key"], key_ranking_sum[nt_interval - 1] / (run + 1)))

        guessing_entropy = key_ranking_sum / runs #this returns GE in an array (increasing traces => no of steps)
        success_rate = success_rate_sum / runs   #this returns SR in an array (increasing traces => no of steps)

        return guessing_entropy, success_rate, key_probabilities_key_ranks
