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
        I_resized, bb_scaled, _ = prepare_result

        I_padded, h_pad, w_pad = pad_image_with_zeros(I_resized)
        S_hat, features_hat = prepare_S_hat_and_features_hat(self.S_mean_centered, self.S_mean_centered, self.features_mean, bb_scaled, w_pad, h_pad)

        return I_padded, S_hat, features_hat
      
    def _prepare_image_for_prediction_with_S_true(self, I_file_path, annotation_folder_path):
        prepare_result = prepare_image_and_bounding_box(I_file_path, self.bb_target_size)
        if prepare_result is None:
            return None
        I_resized, bb_scaled, bb_scale_factor = prepare_result

        I_padded, h_pad, w_pad = pad_image_with_zeros(I_resized)
        S_hat, features_hat = prepare_S_hat_and_features_hat(self.S_mean_centered, self.S_mean_centered, self.features_mean, bb_scaled, w_pad, h_pad)

        I_id = I_file_path.split('/')[-1].replace('.jpg', '')
        image_to_annotation_dict = build_image_to_annotation_dict(annotation_folder_path)
        S_true = scale_S_true_to_bb_and_pad(I_id, annotation_folder_path, image_to_annotation_dict, bb_scale_factor, w_pad, h_pad)

        return I_padded, S_hat, features_hat, S_true

    def some_error_thing(self, I_file_path, annotation_folder_path):
        prepare_result = self._prepare_image_for_prediction_with_S_true(I_file_path, annotation_folder_path)
        if prepare_result is None:
            return None
        else:
            I_padded, S_hat, features_hat, S_true = prepare_result
            #TODO Do stuff here ...

    # Compute average landmark distance from the ground truth landamarks normalized by the distance between eyes for a single image.
    def compute_error(self, S_hat, S_true):
        interocular_distance = np.abs(np.linalg.norm(S_true[153]-S_true[114]))
        average_distance = np.abs(np.linalg.norm(S_hat - S_true, axis=-1)/interocular_distance)
        return average_distance.mean()

    #TODO: S_true array needs to be updated
    def compute_error_all(self, I_file_path, annotation_folder_path, model):
        test_names = get_all_file_names(I_file_path)
        S_hat = []
        S_true = []
        for index in len(test_names):
            file = test_names[index]
            _, S_hat_predicted, _ = model.predict(I_file_path + file)
            S_hat.append(S_hat_predicted)
            
        return self.compute_error(S_hat, S_true).mean()
