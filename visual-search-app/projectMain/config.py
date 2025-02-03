import os

def image_data_with_features_pkl(client_id, model_name="vgg19"):
    """Ensure the directory exists and return the path for the metadata (.pkl) file."""
    folder = os.path.join('metadata-files', f'{client_id}_{model_name}')
    os.makedirs(folder, exist_ok=True) 
    return os.path.join(folder, f'{client_id}_{model_name}_image_data_features.pkl')

def image_features_vectors_idx(client_id, model_name="vgg19"):
    """Ensure the directory exists and return the path for the FAISS index (.idx) file."""
    folder = os.path.join('metadata-files', f'{client_id}_{model_name}')
    os.makedirs(folder, exist_ok=True)  
    return os.path.join(folder, f'{client_id}_{model_name}_image_features_vectors.idx')
