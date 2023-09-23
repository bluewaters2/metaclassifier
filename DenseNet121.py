import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
import h5py
from dataprocess import dataset_precess
# Load the Inception-ResNet v1 model pre-trained on ImageNet data
def DenseNet121(dataset):
  for image in range(0, len(dataset)):
    model = DenseNet121(weights='imagenet', include_top=False, input_tensor=Input(shape=(299,299,3)))  # Exclude the top classification layer
    model = Model(inputs=model.input, outputs=model.get_layer('conv_5b').output)
    image_size = (224, 224)

    features = []
    labels = []
    count = 1;
    for image_path in range(0, len(list_files)):
        #print ("[INFO] Processing - " + str(count) + " named " + list_files[image_path])
        img = load_img(cur_path + "/" + list_files[image_path], target_size=image_size)
        x = img_to_array(img) 
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        flat = feature.flatten()
        features.append(flat)
        labels.append(label)
        count += 1
    # encode the labels using LabelEncoder
    le = LabelEncoder()
    le_labels = le.fit_transform(labels)
    
    try:
        h5f_data = h5py.File(features_path, 'w')
    except:
        a=1;
        
    h5f_data.create_dataset('dataset_1', data=np.array(features))
    
    h5f_label = h5py.File(labels_path, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(le_labels))
    
    h5f_data.close()
    h5f_label.close()
    
    # save model and weights
    model_json = model.to_json()
    with open(model_path + str(test_size) + ".json", "w") as json_file:
      json_file.write(model_json)
    # save weights
    model.save_weights(model_path + str(test_size) + ".h5")
    
    print ("Features saved")
