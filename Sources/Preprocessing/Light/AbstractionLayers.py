import numpy as np
import colorsys
from scipy.interpolate import interp1d


def resolve_group(group, luminaire_data):
    positional_info_key = group["Position_Info"]
    fixture_type_key = group["Fixture_Type"]
    Ids = []
    for id, attributes in luminaire_data.items():
        if attributes.get('Positional Information') == positional_info_key and attributes.get(
                'Fixture Type') == fixture_type_key:
            Ids.append(id)
    group["IDs"] = Ids
    assert len(Ids), "Resolving Group did not work, please check Group definition!"


class AbstractionLayer:
    def __init__(self):
        pass

    def abstract_data(self, data, abstraction_config):
        raise NotImplementedError()

    def get_size(self):
        raise NotImplementedError()

    def get_brightness(self, data):
        raise NotImplementedError()

    def get_hue(self, data):
        raise NotImplementedError()

    def get_sat(self, data):
        raise NotImplementedError()

    def get_pan(self, data):
        raise NotImplementedError()

    def get_tilt(self, data):
        raise NotImplementedError()

    def get_attribute_sizes(self):
        raise NotImplementedError()  # brightness, hue, sat, pan, tilt size

    """
        Overwrite brightness values of data with new_data
    """

    def set_brightness(self, data, new_data):
        raise NotImplementedError()

    """
        Overwrite hue values of data with new_data
    """

    def set_hue(self, data, new_data):
        raise NotImplementedError()

    """
        Overwrite sat values of data with new_data
    """

    def set_sat(self, data, new_data):
        raise NotImplementedError()

    """
            Overwrite sat values of data with new_data
        """

    def set_pan(self, data, new_data):
        raise NotImplementedError()

    """
            Overwrite sat values of data with new_data
        """

    def set_tilt(self, data, new_data):
        raise NotImplementedError()

class PASv01(AbstractionLayer):
    def __init__(self):
        self.size = 60
        self.ip_idx = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54] # Intensity Moving Head
        self.ip_idx_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]
        self.pd_idx = [x+2 for x in self.ip_idx] # AF Peak Density
        self.int_idx = [] # Intensity HSV Fixture
        self.int_idx_names = []
        self.attribute_sizes = (10, 10, 10, 0, 0) # Wie groß sind die Arrays die ich zurück bekomme # brightness, hue, sat, pan, tilt size
        self.hue_idx = [4, 10, 16, 22, 28, 34, 40, 46, 52, 58]
        self.sat_idx = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
        self.col_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]
        self.pan_idx = []
        self.tilt_idx = []
        self.mov_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]
        self.tilt_het_idx=[]

    # interface functions

    def get_size(self):
        return self.size

    def get_brightness(self, data):
        names = self.ip_idx_names + self.int_idx_names
        ip_data = data[:,self.ip_idx].copy()

        #multiplying with peak density
        pd_data = data[:, self.pd_idx]
        res = np.multiply(ip_data, pd_data)

        int_data = data[:,self.int_idx].copy()
        return np.concatenate((res, int_data), axis=1), names  # data and attribute names

    def get_hue(self, data):
        return data[:,self.hue_idx].copy(), self.col_names

    def get_sat(self, data):
        return data[:,self.sat_idx].copy(), self.col_names

    def get_pan(self, data):
        return data[:,self.pan_idx].copy(), self.mov_names

    def get_tilt(self, data):
        return data[:,self.tilt_idx].copy(), self.mov_names

    def get_attribute_sizes(self):
        return self.attribute_sizes  # brightness, hue, sat, pan, tilt size

    def set_brightness(self, data, new_data):
        idx = self.ip_idx + self.int_idx
        data[:,idx] = new_data
        pd_data = data[:,self.pd_idx]
        pd_data.fill(1)
        data[:, self.pd_idx] = pd_data
        return data

    def set_hue(self, data, new_data):
        data[:,self.hue_idx] = new_data
        return data

    def set_sat(self, data, new_data):
        data[:,self.sat_idx] = new_data
        return data

    def set_pan(self, data, new_data):
        data[:,self.pan_idx] = new_data
        return data

    def set_tilt(self, data, new_data):
        data[:,self.tilt_idx] = new_data
        tilt_het_data = data[:,self.tilt_het_idx]
        tilt_het_data.fill(0)
        data[:, self.tilt_het_idx] = tilt_het_data
        return data

    def abstract_data(self, data, abstraction_config):

        to_be_extracted_groups = abstraction_config['to be extracted groups']

        for key, subdict in data.items():
            if 'BRIGHTNESS' in subdict:
                brightness_length = len(subdict['BRIGHTNESS'])
                print(brightness_length)  # Print the length of the 'BRIGHTNESS' list
                break  # Exit the loop after finding the first 'BRIGHTNESS' list

        al_combined_data_list = []

        for tbe_group in to_be_extracted_groups:
            # print(group)
            al_extract_function = tbe_group['AL_mode']
            luminaire_group = tbe_group['Group']
            group_extract = self.extract_group(data, abstraction_config['Group_definitions'][luminaire_group])
            if al_extract_function == 'pairsearchv1':
                al_data = self.calc_abs_layer_pairsearchv1(group_extract)
            else:
                al_data = np.array([])

            al_data = al_data.T

            # Check if al_data is not empty and append to list
            if al_data.size > 0:
                al_combined_data_list.append(al_data)

        if al_combined_data_list:  # Check if the list is not empty
            al_combined_data = np.concatenate(al_combined_data_list, axis=1)
        else:
            print('al list is empty')
            al_combined_data = np.array([])

        return al_combined_data

    #######################

    def bound(self, x, bl, bu):
        # return bounded value clipped between bl and bu
        y = min(max(x, bl), bu)
        return y

    #######################

    def get_mean_vel(self, group):
        len_group = group.shape[0]
        meanvel = np.zeros(len_group)

        for i in range(len_group):
            meanvel[i] = np.mean(group[i, :])

        return meanvel

    #######################

    def get_tilt_values_large(self, tilt_group):
        len = tilt_group.shape[0]
        tilt_max_val = np.zeros(len)
        tilt_min_val = np.zeros(len)
        maxpos = np.zeros(len, dtype=int)
        minpos = np.zeros(len, dtype=int)

        # Maxima absolute
        for i in range(len):
            maxpos[i] = np.argmax(tilt_group[i])

        # Minima absolute
        for i in range(len):
            minpos[i] = np.argmin(tilt_group[i])

        # Get the values
        for i in range(len):
            tilt_min_val[i] = tilt_group[i, minpos[i]]
            tilt_max_val[i] = tilt_group[i, maxpos[i]]

        return tilt_max_val, tilt_min_val

    #######################

    def get_pan_values_large(self, pan_group):
        len = pan_group.shape[0]
        width = pan_group.shape[1]

        pan_left_max_pos = np.zeros(len)
        pan_left_max_val = np.zeros(len)
        pan_right_max_pos = np.zeros(len)
        pan_right_max_val = np.zeros(len)

        maxvel = np.zeros((len, 1))
        maxpos = np.zeros((len, 1))
        maxposrel = np.zeros((len, 1))

        # Maxima absolute
        for i in range(len):
            maxvel[i], maxpos[i] = np.max(pan_group[i, :]), np.argmax(pan_group[i, :])
            maxposrel[i] = maxpos[i] / width

        minvel = np.zeros((len, 1))
        minpos = np.zeros((len, 1))
        minposrel = np.zeros((len, 1))

        # Minima absolute
        for i in range(len):
            minvel[i], minpos[i] = np.min(pan_group[i, :]), np.argmin(pan_group[i, :])
            minposrel[i] = minpos[i] / width

        # Get the values
        for i in range(len):
            maxpos_temp = int(maxpos[i])
            minpos_temp = int(minpos[i])
            #
            pan_left_max_pos[i] = minposrel[i]
            pan_left_max_val[i] = pan_group[i, minpos_temp]
            pan_right_max_pos[i] = maxposrel[i]
            pan_right_max_val[i] = pan_group[i, maxpos_temp]

        return pan_left_max_pos, pan_left_max_val, pan_right_max_pos, pan_right_max_val

    #######################

    def get_colhue_colsat_at_pos(self, colhue_group, colsat_group, dim):
        len = colhue_group.shape[0]

        colhue_lead = np.zeros(len)
        colsat_lead = np.zeros(len)
        colhue_back = np.zeros(len)
        colsat_back = np.zeros(len)

        maxvel = np.zeros((len, 1))
        maxpos = np.zeros((len, 1))

        # Maxima absolute
        for i in range(len):
            maxvel[i], maxpos[i] = np.max(dim[i, :]), np.argmax(dim[i, :])

        minvel = np.zeros((len, 1))
        minpos = np.zeros((len, 1))

        # Minima absolute
        for i in range(len):
            minvel[i], minpos[i] = np.min(dim[i, :]), np.argmin(dim[i, :])

        # get color max from brightest fixture in the group and vice versa for the darkest
        for i in range(len):
            maxpos_temp = int(maxpos[i])
            minpos_temp = int(minpos[i])
            #
            colhue_lead[i] = colhue_group[i, maxpos_temp]
            colsat_lead[i] = colsat_group[i, maxpos_temp]
            colhue_back[i] = colhue_group[i, minpos_temp]
            colsat_back[i] = colsat_group[i, minpos_temp]

        return colhue_lead, colsat_lead, colhue_back, colsat_back

    #######################

    def find_maxvel_maxpos(self, group):
        len, width = group.shape
        maxvel = np.zeros((len, 1))
        maxpos = np.zeros((len, 1))

        # Maxima relative
        for i in range(len):
            maxvel[i], maxpos[i] = np.max(group[i, :]), np.argmax(group[i, :])
            maxpos[i] = maxpos[i] / width

        return maxvel, maxpos

    #######################

    def find_minvel_minpos(self, group):
        len, width = group.shape
        minvel = np.zeros((len, 1))
        minpos = np.zeros((len, 1))

        # Minima relative
        for i in range(len):
            minvel[i], minpos[i] = np.min(group[i, :]), np.argmin(group[i, :])
            minpos[i] = minpos[i] / width

        return minvel, minpos

    #######################

    def get_alternating_factors(self, group):
        len, width = group.shape

        # Feature Extractor Alternating Factors
        # 1 = density_peaks, 2 = peak_simularity,
        # 3 = Pos 2nd Peak (if exists), 4 = Intensity of Minima absolute,
        # 5 = Minima Simularity
        density_peaks = np.zeros(len)
        peak_simularity = np.zeros(len)
        minima_simularity = np.zeros(len)
        alternating_factors = np.zeros((len, 5))

        minvel, _ = self.find_minvel_minpos(group)
        # gradients = get_gradient(group)
        # print(group.shape)

        for i in range(len):
            # print(group)
            val_peaks, pos_peaks = self.find_peaks_np(group[i, :])
            # print(pos_peaks)
            peaks_len = val_peaks.size
            # print(peaks_len)
            size_pos_peaks = pos_peaks.size
            # print(val_peaks)
            density_peaks[i] = size_pos_peaks / width
            # print(density_peaks[i])
            # for the case that all values are the same the density will be 1. CHECK!?!?
            if peaks_len == 0:
                density_peaks[i] = 1
            alternating_factors[i, 0] = density_peaks[i]

            if peaks_len == 0:
                peak_simularity[i] = 1
            else:
                max_peak = np.max(val_peaks)
                min_peak = np.min(val_peaks)
                peak_simularity[i] = 1 - (max_peak - min_peak)

            alternating_factors[i, 1] = peak_simularity[i]
            # print(peak_simularity)

            if pos_peaks.size > 1:
                relative_pos_2nd_peak = pos_peaks[1] / width
                alternating_factors[i, 2] = relative_pos_2nd_peak
            elif pos_peaks.size == 0:
                alternating_factors[i, 2] = 0
            else:
                alternating_factors[i, 2] = pos_peaks[0] / width
            # print(alternating_factors[i, 2])

            alternating_factors[i, 3] = minvel[i]
            inverted_group = 1 - group[i, :]
            val_minima, _ = self.find_peaks_np(inverted_group)
            minima_len = val_minima.size

            if minima_len == 0:
                minima_simularity[i] = 1
            else:
                max_minima = np.max(val_minima)
                min_minima = np.min(val_minima)
                minima_simularity[i] = 1 - (max_minima - min_minima)

            alternating_factors[i, 4] = minima_simularity[i]
            # print(alternating_factors[i, 4])
            # print(alternating_factors)

        return alternating_factors

    #######################

    def find_peaks_np(self, array):
        ### Is it ok give back an empty array in case of no existing peaks?

        peak_indices = []
        peak_values = []
        same_values = (array[0] == array).all()

        # If the entire array has the same values, return empty lists
        if same_values:
            return np.array(peak_values), np.array(peak_indices)

        # Check if the first element is the highest
        if array[0] > array[1]:
            # print('first')
            peak_indices.append(0)
            peak_values.append(array[0])

        # Slide along the array and check if the current element is greater than to its neighbors
        for i in range(1, len(array) - 1):
            if array[i] > array[i - 1] and array[i] > array[i + 1]:
                # print('mid')
                # print(i)
                peak_indices.append(i)
                peak_values.append(array[i])

        """
        # Slide along the array and check if the current peak element is equal to its neighbors
        if len(peak_indices) != 0:
            for i in range(peak_indices[0], (peak_indices[-1] - peak_indices[0] + 1)):
                # print(i)
                #if array[i] == array[i - 1] and array[i] == array[i + 1]:
                if array[i] == array[i - 1] and array[peak_indices[0]] == array[i]:
                    # print(i)
                    peak_indices.append(i)
                    peak_values.append(array[i])
        """

        # Check if the last element is the highest
        if array[-1] > array[-2]:
            # print('last')
            peak_indices.append(len(array) - 1)
            peak_values.append(array[-1])

        # print(peak_indices)

        return np.array(peak_values), np.array(peak_indices)

    #######################

    def get_gradient(self, group):
        len, width = group.shape
        gradients = np.zeros((len, 1))

        maxvel = np.zeros((len, 1))
        maxpos = np.zeros((len, 1))

        # Maxima absolute
        for i in range(len):
            maxvel[i], maxpos[i] = np.max(group[i, :]), np.argmax(group[i, :])

        minvel = np.zeros((len, 1))
        minpos = np.zeros((len, 1))

        # Minima absolute
        for i in range(len):
            minvel[i], minpos[i] = np.min(group[i, :]), np.argmin(group[i, :])

        for i in range(len):
            gradient_segment = np.zeros((1, width + 4))
            gradient_segment[0, 2:width + 2] = group[i, :]
            mean_temp = np.gradient(gradient_segment, axis=1)
            mean_temp = np.abs(mean_temp)
            nonzero_indices = np.nonzero(mean_temp)
            mean_temp = mean_temp[nonzero_indices]

            mean_temp_calc = np.zeros((1, mean_temp.size))
            mean_temp_calc = mean_temp
            if mean_temp.size == 0:
                mean_temp_calc = np.zeros((1, 1))

            # CHECK the *2 factor in the visualizer ?!?!?
            gradients[i] = self.bound((np.mean(mean_temp_calc) * 2), 0, 1)

        return gradients

    #######################

    def extract_group(self, luminaire_data, group):
        extracted_values = []
        if "IDs" not in group.keys():
            resolve_group(group, luminaire_data)
        for id in group["IDs"]:
            extracted_values.append(luminaire_data[id])
        return extracted_values

    #######################

    def extract_values(self, luminaire_data, positional_info_key, fixture_type_key, data_key):
        extracted_values = []

        for _, attributes in luminaire_data.items():
            if attributes.get('Positional Information') == positional_info_key and attributes.get(
                    'Fixture Type') == fixture_type_key:
                key_list = attributes.get(data_key, [])
                if key_list:  # Check if the list is not empty
                    extracted_values.append(key_list)

        return np.array(extracted_values)

    #######################

    def calc_abs_layer_pairsearchv1(self, group):

        # Target AL Extractor Standard
        # 01 = Intensity of Peak absolute
        # 02 = Slope of Peak Intensity
        # 03 = AF Peak Density
        # 04 = AF Peak Simularity
        # 05 = Col Hue Mean
        # 06 = Col Sat Mean

        brightness_values = np.array([luminaire['BRIGHTNESS'] for luminaire in group if 'BRIGHTNESS' in luminaire])
        length = brightness_values.shape[1]
        al_pairsearch = np.zeros((6, length))

        n = len(next((luminaire['HUE'] for luminaire in group if 'HUE' in luminaire), [0.0] * length))
        colhue_values = np.array([luminaire.get('HUE', [0.0] * n) for luminaire in group])

        n = len(next((luminaire['SATURATION'] for luminaire in group if 'SATURATION' in luminaire), [0.0] * length))
        colsat_values = np.array([luminaire.get('SATURATION', [0.0] * n) for luminaire in group])

        maxvel_brightness, maxpos_brightness = self.find_maxvel_maxpos(brightness_values.T)
        al_pairsearch[0, :] = maxvel_brightness.T
        al_pairsearch[1, :] = (self.get_gradient(brightness_values.T)).T
        alternating_factors_brightness = self.get_alternating_factors(brightness_values.T)
        al_pairsearch[2, :] = (alternating_factors_brightness[:, 0]).T
        al_pairsearch[3, :] = (alternating_factors_brightness[:, 1]).T
        al_pairsearch[4, :] = (self.get_mean_vel(colhue_values.T)).T
        al_pairsearch[5, :] = (self.get_mean_vel(colsat_values.T)).T

        return al_pairsearch

    #######################


def get_abstraction_layer(name):
    if name == 'PASv01':
        return PASv01()

    ## Currently PASv02 is feeding into the same pipeline than PASv01
    if name == 'PASv02':
        return PASv01()

    raise NotImplementedError()