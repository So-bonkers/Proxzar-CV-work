import os

def image_data_with_features_pkl(client_id, model_name="vgg19"):
    """Return the path for saving the image data features file with client ID prefix."""
    return os.path.join('metadata-files/', f'{client_id}_{model_name}_image_data_features.pkl')

def image_features_vectors_idx(client_id, model_name="vgg19"):
    """Return the path for saving the FAISS index file with client ID prefix."""
    return os.path.join('metadata-files/', f'{client_id}_{model_name}_image_features_vectors.idx')
