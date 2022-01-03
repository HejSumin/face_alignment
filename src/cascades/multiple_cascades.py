from src.face_alignment.utility import *
from src.cascades.single_cascade import *

class MultipleCascades(): #TODO Dataclass

    def __init__(self, cascades, S_mean_centered, features_mean):
        self.cascades = cascades
        self.S_mean_centered = S_mean_centered
        self.features_mean = features_mean
        self.bb_target_size = 500 # TODO make parameter

    def predict(self, I_file_path):
        prepare_result = self._prepare_image_for_prediction(I_file_path)
        if prepare_result is None:
            return None
        else:
            I_padded, S_hat, features_hat = prepare_result
            
            for cascade in self.cascades:
                S_hat_new, features_hat_new = cascade.apply_cascade(I_padded, S_hat, features_hat, self.S_mean_centered, self.features_mean)
                S_hat = S_hat_new
                features_hat = features_hat_new

            return I_padded, S_hat, features_hat

    def _prepare_image_for_prediction(self, I_file_path):
        prepare_result = prepare_image_and_bounding_box(I_file_path, self.bb_target_size)
        if prepare_result is None:
            return None
        I_resized, bb_scaled, bb_scale_factor = prepare_result

        I_padded, h_pad, w_pad = pad_image_with_zeros(I_resized)
        S_hat, features_hat = prepare_S_hat_and_features_hat(self.S_mean_centered, self.S_mean_centered, self.features_mean, bb_scaled, w_pad, h_pad)

        return I_padded, S_hat, features_hat
