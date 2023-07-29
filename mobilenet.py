import cv2
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.applications as applications
import tensorflow.keras.preprocessing.image as image


from sklearn.metrics import confusion_matrix

class_mapping = {
    1: 'Penguin',
    2: 'Turtle',
    3: 'Background'
}

def run_mobilenet(training_df : pd.DataFrame,
                  validation_df : pd.DataFrame, ) -> None:
    
    # get truth and image data
    truth = np.array(training_df['category_id'].tolist())

    # load and preprocess data
    glob_dir = "images/training/*.jpg"
    images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)]
    data = np.array(np.float32(images)/255)
    print(data)
    return

    # set up model
    base_model = applications.MobileNetV2((224, 224, 3), 
                                          include_top = False,
                                          weights = None)
    base_model.trainable = False
    

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    predictions = model.predict(data)
    prediction_labels = np.argmax(predictions, axis=1)
    print(prediction_labels)


    '''
    # compute confustion matric
    cm = confusion_matrix(truth, predicted_labels, labels=[0, 1, 2])

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("MobileNet Confustion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
'''