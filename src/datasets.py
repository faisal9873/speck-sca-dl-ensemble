
class SCADatasets:

    def __init__(self):
        self.trace_set_list = []

    def get_trace_set(self, trace_set_name):
        trace_list = self.get_trace_set_list()
        return trace_list[trace_set_name]

    def get_trace_set_list(self):  #ASCAD.h5 #aes_hd.h5
        parameters_speck_fixed_key = {
            "file": "SPECK_fixed_data.h5", #sorted
            "key": "1122334455667788",    #sorted
            "key_offset": 32,
            "input_offset": 0,  #sorted
            "data_length": 32,  #sorted 
            "first_sample": 0,  #sorted
            "number_of_samples": 5000, #5000,  #sorted
            "n_profiling": 8000,   #sorted 60000
            "n_attack": 1000,   #sorted 7000
            "classes": 9,   #sorted
            "good_key": 136, #0x7788, [119  136]
            "number_of_key_hypothesis": 256,
            "epochs": 50,   #sorted
            "mini-batch": 50    #sorted
        }

        parameters_speck_variable_key = {
            "file": "SPECK_variable_data_masked.h5",
            "key": "1122334455667788",
            "key_offset": 32,
            "input_offset": 0,  #sorted
            "data_length": 32,  #sorted
            "first_sample": 0,  #sorted
            "number_of_samples": 5000,  #sorted
            "n_profiling": 80000,  #sorted
            "n_attack": 20000,  #sorted
            "classes": 9,   #sorted
            "good_key": 136, #0x7788, [119  136]
            "number_of_key_hypothesis": 256,
            "epochs": 50,   #sorted
            "mini-batch": 400   #sorted
        }

        parameters_ches_ctf = {
            "file": "ches_ctf.h5",
            "key": "2EEE5E799D72591C4F4C10D8287F397A",
            "key_offset": 32,
            "input_offset": 0,
            "data_length": 48,
            "first_sample": 0,
            "number_of_samples": 2200,
            "n_profiling": 45000,
            "n_attack": 5000,
            "classes": 9,
            "good_key": 46,
            "number_of_key_hypothesis": 256,
            "epochs": 50,
            "mini-batch": 400
        }

        self.trace_set_list = {
            "SPECK_fixed_key": parameters_speck_fixed_key,
            "SPECK_variable_key": parameters_speck_variable_key,
            "ches_ctf": parameters_ches_ctf
        }

        return self.trace_set_list
