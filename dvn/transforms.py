import torch
# tranforms


class Normalize(object):
    """Normalizes keypoints.
    """
    def __init__(self, max_val, min_val, new_max, new_min):
        self.max = max_val
        self.min = min_val
        self.new_max = new_max
        self.new_min = new_min
    
    def __call__(self, sample):

        image = (sample - self.min) * (self.new_max - self.new_min)/(self.max - self.min) + self.new_min
        ##############################################################
        # End of your code                                           #
        ##############################################################
        return image

