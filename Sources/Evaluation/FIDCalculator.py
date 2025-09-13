# example of calculating the frechet inception distance
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm


class FIDCalculator:
    def __init__(self, model_config):
        self.model_config = model_config

# calculate frechet inception distance
    def calculate_fid(self, act1, act2):
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def process(self, predictions, test_data):
        return self.calc_pfid(predictions, test_data), self.calc_sfid(predictions, test_data)

    # Fid von Positionen
    def calc_pfid(self, predictions, test_data):
        pr_features = np.empty((0, predictions[0].shape[1]))
        t_features = np.empty((0, predictions[0].shape[1]))

        for i in range(len(predictions)):
            pr_features = np.concatenate((pr_features, np.array(predictions[i])))
            t_features = np.concatenate((t_features, np.array(test_data[i]['lighting_array'])))

        return self.calculate_fid(np.array(pr_features), np.array(t_features))

    # Fid von Geschwindigkeiten in den Parametern
    def calc_sfid(self, predictions, test_data):
        pr_features = np.empty((0, predictions[0].shape[1]))
        t_features = np.empty((0, predictions[0].shape[1]))

        for i in range(len(predictions)):
            p = np.array(predictions[i])
            p_1 = p[1:]
            p = p[:-1]
            p_d = p_1 - p

            pr_features = np.concatenate((pr_features, p_d))

            t = np.array(test_data[i]['lighting_array'])
            t_1 = t[1:]
            t = t[:-1]
            t_d = t_1 - t

            t_features = np.concatenate((t_features, t_d))

        return self.calculate_fid(np.array(pr_features), np.array(t_features))

