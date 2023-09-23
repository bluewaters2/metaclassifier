import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from metaclassifier import metaclassifier

def process():
  input_root = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/MyDrive/lung_image_sets', labels='inferred')
  datasets = input_root.class_names
  datasets
def check_if_directory_exists(name_folder):
    if not os.path.exists(name_folder):
        print(name_folder + " directory does not exist, created")
        os.makedirs(name_folder) 
    else:
        print(name_folder + " directory exists, no action performed")

def load_json_file(name_json):
    """
    load_json_file(name_json)
    INPUT:
        name_json: name of the json file to be loaded
    OUTPUT:
        variable with the json file loaded
        
    @author: Eduardo Fidalgo (EFF)
    """
    import json
    with open(name_json) as f:    
      config = json.load(f)
      
    return config
  def read_image_names(path_ima='./images',img_ext='*.png',verbose=False):

    import os
    import glob

    full_names_img = list()
    prev_path = os.getcwd()
    full_names_img = glob.glob(os.path.join(path_ima,img_ext))

    if verbose:
        print('(3) The "{0}" directory has {1} images with {2} extension.\n'
              .format(os.getcwd(),len(full_names_img),img_ext))

    return full_names_img

def output(metaclassifier.h5):
  f = open(results, "w")
  rank_1 = 0
  rank_5 = 0

  for (label, features) in zip(testLabels, testData):
  # take the top-5 class labels
    predictions = model.predict_proba(np.atleast_2d(features))[0]
    predictions = np.argsort(predictions)[::-1][:5]

    if label == predictions[0]:
      rank_1 += 1

    if label in predictions:
      rank_5 += 1

  rank_1 = (rank_1 / float(len(testLabels))) * 100
  rank_5 = (rank_5 / float(len(testLabels))) * 100

  f.write("Rank-1: {:.2f}%\n".format(rank_1))
  f.write("Rank-5: {:.2f}%\n\n".format(rank_5))
# evaluate the model of test data
  preds = model.predict(testData)

# write the classification report to file
  f.write("{}\n".format(classification_report(testLabels, preds)))
  f.close()

# dump classifier to file
  print ("[INFO] saving model...")
  pickle.dump(model, open(classifier_path, 'wb'))

# display the confusion matrix
  print ("[INFO] confusion matrix")

# get the list of training lables
  labels = sorted(list(os.listdir(train_path)))

# plot the confusion matrix
  cm = confusion_matrix(testLabels, preds)
  sns.heatmap(cm,
            annot=True,
            cmap="Set2")
  plt.show()
