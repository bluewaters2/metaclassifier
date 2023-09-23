import h5py
import numpy as np

# Load feature vectors from each file
file_paths = ['InceptionResNetv1_features.h5', 'Efficinetb7_features.h5', 'DenseNet121_features.h5']

all_features = []

for file_path in file_paths:
    with h5py.File(file_path, 'r') as hf:
        features = hf['inception_resnet_features'][:]
        all_features.append(features)

# Concatenate the feature vectors
concatenated_features = np.concatenate(all_features, axis=1)

# Save the concatenated features to a new file
output_file = 'concatenated_features.h5'
with h5py.File(output_file, 'w') as hf:
    hf.create_dataset('concatenated_features', data=concatenated_features)

print("Features concatenated")
