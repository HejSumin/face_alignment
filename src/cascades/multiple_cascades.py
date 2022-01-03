from src.face_alignment.utility import *

class MultipleCascades(): #TODO Dataclass

    def __init__(self, cascades, S_mean, features_mean):
        self.cascades = cascades
        self.S_mean = S_mean
        self.features_mean = features_mean

    def predict(self, I_file_path):
        #prepare_image_for_prediction

        return

    def prepare_image_for_prediction(self, I_file_path):
        prepare_result = prepare_image_and_bounding_box(I_file_path)
        if prepare_result is None:
            return None
        I_resized, bb_scaled, bb_scale_factor = prepare_result

        I_padded, h_pad, w_pad = pad_image_with_zeros(I_resized)

        return

    def error_function():
        return