import json
import numpy as np



path = "/new_data/Lighting_JSONs/User1/ID02MR02_features_v01_1800_untouched.json"

if __name__ == '__main__':
    with open(path) as f:
        sample_dict = json.loads(f.read())
        np_light_seq = np.array(sample_dict['lighting_array'])
        test_data = np_light_seq[:,0]
        print("done")


