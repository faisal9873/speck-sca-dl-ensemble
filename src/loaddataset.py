import numpy as np  #imports
import h5py


class LoadDatasets:

    def speck_labelize(self, trace_data, byte, leakage_model):

        pt_ct = [int(''.join(f'{int(byte):02X}' for byte in row[0:4]), 16) for row in trace_data]

        # Extracts the last two bytes from each row and combines them into a single integer
        key_byte = [int(''.join(f'{int(byte):02X}' for byte in row[32:40]), 16) for row in trace_data]

        key_byte = np.asarray(key_byte[:])                      # Convert key to array



        # SPECK uses a 32-bit block, so split plaintext into two 16-bit words
        intermediate_values = []
        for pt, k in zip(pt_ct, key_byte):
            # Split 32-bit plaintext block into two 16-bit words
            y = pt & 0xFFFF        # Lower 16 bits
            x = (pt >> 16) & 0xFFFF  # Upper 16 bits
            k = int(k) & 0xFFFF
    
            # Perform one round of SPECK encryption
            x = ((x << 7) | (x >> 9)) & 0xFFFF  # Rotate x left by 7 bits
            x = (x + y) & 0xFFFF                # Addition mod 2^16aes
            x ^= k                              # XOR with the round key
            y = ((y >> 2) | (y << 14)) & 0xFFFF # Rotate y right by 2 bits
            y ^= x                              # XOR with x
    
            # Combine y and x into a single 32-bit value for the intermediate state
            intermediate_value = (x << 16) | y
            
            # Extract only the lower byte (last 8 bits)
            lower_byte = intermediate_value & 0xFF  # This isolates 0x88 if intermediate_value is 0x7788
            # upper_byte = (intermediate_value >> 8) & 0xFF  # This isolates 0x77 if intermediate_value is 0x7788
            intermediate_values.append(lower_byte)
            # intermediate_values.append(upper_byte)
            
        
        
        # Convert intermediate values to Hamming Weight if using HW leakage model
        # if leakage_model == "HW":
        #     return [bin(iv).count("1") for iv in intermediate_values]
        if leakage_model == "HW":
            hw_values = [bin(iv).count("1") for iv in intermediate_values]
            # print(f'Last element HW: {hw_values[-1]}')
            return hw_values
        else:
            return intermediate_values
        

    def load_dataset(self, dataset_file, n_profiling, n_attack, target_byte, leakage_model):

        if "ches_ctf.h5" in dataset_file:
            in_file = h5py.File(dataset_file, 'r')
            profiling_samples = np.array(in_file.get('profiling_traces'))
            profiling_data = np.array(in_file.get('profiling_data'))
            attack_samples = np.array(in_file.get('attacking_traces'))
            attack_data = np.array(in_file.get('attacking_data'))
        else:
            in_file = h5py.File(dataset_file, "r")
            profiling_samples = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
            attack_samples = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
            profiling_plaintext = in_file['Profiling_traces/metadata']['plaintext']
            attack_plaintext = in_file['Attack_traces/metadata']['plaintext']
            profiling_key = in_file['Profiling_traces/metadata']['key']
            attack_key = in_file['Attack_traces/metadata']['key']
            profiling_data = np.zeros((n_profiling, 48))
            attack_data = np.zeros((n_profiling, 48))
            for i in range(n_profiling):
                profiling_data[i][0:4] = profiling_plaintext[i]
                profiling_data[i][32:40] = profiling_key[i]
            for i in range(n_attack):
                attack_data[i][0:4] = attack_plaintext[i]
                attack_data[i][32:40] = attack_key[i]

        nt = n_profiling #number of profiling traces
        na = n_attack    #number of attack traces

        X_profiling = profiling_samples[0:nt]
        Y_profiling = self.speck_labelize(profiling_data[0:nt], target_byte, leakage_model)


        X_attack = attack_samples[0:na]
        Y_attack = self.speck_labelize(attack_data[0:na], target_byte, leakage_model)



        # attack set is split into validation and attack sets.
        X_validation = X_attack[0: int(na / 2)]
        Y_validation = Y_attack[0: int(na / 2)]
        X_attack = X_attack[int(na / 2): na]
        Y_attack = Y_attack[int(na / 2): na]

        profiling_data = profiling_data[0:nt]
        validation_data = attack_data[0: int(na / 2)]
        attack_data = attack_data[int(na / 2): na]

        return (X_profiling, Y_profiling), (X_validation, Y_validation), (X_attack, Y_attack), (
            profiling_data, validation_data, attack_data)


