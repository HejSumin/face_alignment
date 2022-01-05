from src.face_alignment.utility import *
from src.cascades.single_cascade import *

class MultipleCascades():

    def __init__(self, cascades, S_mean_centered, features_mean, is_averaging_mode, averaging_tree_amount):
        self.cascades = cascades
        self.S_mean_centered = S_mean_centered
        self.features_mean = features_mean
        self.is_averaging_mode = is_averaging_mode
        self.averaging_tree_amount = averaging_tree_amount

    def predict(self, I_file_path):
        prepare_result = self._prepare_image_for_prediction(I_file_path)
        if prepare_result is None:
            return None
        else:
            I_padded, S_hat, features_hat = prepare_result
            S_hat, features_hat = self.apply_cascades(I_padded, S_hat, features_hat)

            return I_padded, S_hat, features_hat

    def validate_train_image(self, I_file_path, annotation_folder_path):
        prepare_result = self._prepare_train_image_for_validation_with_S_true(I_file_path, annotation_folder_path)
        if prepare_result is None:
            return None
        else:
            I_padded, S_hat, features_hat, S_true = prepare_result
            S_hat, features_hat = self.apply_cascades(I_padded, S_hat, features_hat)
            
            return I_padded, S_hat, features_hat

    def apply_cascades(self, I_padded, S_hat, features_hat):
        for cascade in self.cascades:

            if self.is_averaging_mode:
                S_hat_new, features_hat_new = cascade.apply_cascade_in_averaging_mode(I_padded, S_hat, features_hat, self.S_mean_centered, self.features_mean, self.averaging_tree_amount)
            else:
                S_hat_new, features_hat_new = cascade.apply_cascade(I_padded, S_hat, features_hat, self.S_mean_centered, self.features_mean)

            S_hat = S_hat_new
            features_hat = features_hat_new

        return S_hat, features_hat

    def _prepare_image_for_prediction(self, I_file_path):
        prepare_result = prepare_image_and_bounding_box(I_file_path)
        if prepare_result is None:
            return None
        I_resized, bb_scaled, _ = prepare_result

        I_padded, h_pad, w_pad = pad_image_with_zeros(I_resized)
        S_hat, features_hat = prepare_S_hat_and_features_hat(self.S_mean_centered, self.S_mean_centered, self.features_mean, bb_scaled, w_pad, h_pad)

        return I_padded, S_hat, features_hat
      
    def _prepare_train_image_for_validation_with_S_true(self, I_file_path, annotation_folder_path):
        prepare_result = prepare_image_and_bounding_box(I_file_path)
        if prepare_result is None:
            return None
        I_resized, bb_scaled, bb_scale_factor = prepare_result

        I_padded, h_pad, w_pad = pad_image_with_zeros(I_resized)
        S_hat, features_hat = prepare_S_hat_and_features_hat(self.S_mean_centered, self.S_mean_centered, self.features_mean, bb_scaled, w_pad, h_pad)

        I_id = I_file_path.split('/')[-1].replace('.jpg', '')
        image_to_annotation_dict = build_image_to_annotation_dict(annotation_folder_path)
        S_true = scale_S_true_to_bb_and_pad(I_id, annotation_folder_path, image_to_annotation_dict, bb_scale_factor, w_pad, h_pad)

        return I_padded, S_hat, features_hat, S_true

    # Compute average landmark distance from the ground truth landamarks normalized by the distance between eyes for a single image.
    def compute_error(S_hat, S_true):
        interocular_distance = np.linalg.norm(S_true[153].astype(np.int32)- S_true[114].astype(np.int32))
        average_distance = np.linalg.norm(S_hat - S_true) / interocular_distance
        return average_distance.mean()

    def compute_error_all(self, I_file_path, annotation_folder_path): #TODO THIS DOES NOT DO WHAT IT SHOULD DO!
        prepare_result = self.validate_train_image(I_file_path, annotation_folder_path) 
        if prepare_result is None:
            return None
        else: 
            _, S_hat, _, S_true = prepare_result

        S_hat_arr = []
        S_true_arr = []
        for index in len(prepare_result):
            S_hat_arr.append(S_hat[index])
            S_true_arr.append(S_true[index])       

        return self.compute_error(S_hat_arr, S_true_arr).mean()
