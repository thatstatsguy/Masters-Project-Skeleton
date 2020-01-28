import glob
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from tensorflow_large_model_support import LMSKerasCallback
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import csv
from keras import callbacks
from keras.models import load_model


def process_image(img_path, targetsize):
    """
    Processes image into target size
    :param img_path: path to image
    :param targetsize: required resolution
    :return: Loaded and transformed image
    """
    img = image.load_img(img_path, target_size=(targetsize, targetsize))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    pImg = preprocess_input(img_array)
    return pImg


def evaluate_model(image_targetsize):
    """
    Model evaluation is performed here on a completely unseen validation set. The reason that the Keras validation
    accuracy is not trusted is due to the fact that Keras has varying definitions for accuracy depending on the
    application. The normal accuracy is calculated here and the outputs of the confusion matrices and predictions
    (with probabilities) are saved to csv.

    See: https://github.com/keras-team/keras/blob/c2e36f369b411ad1d0a40ac096fe35f73b9dffd3/keras/metrics.py#L13 for
    various definitions

    :param image_targetsize: Target image size for image evaluation in line with the design of the network.
    :return:None
    """

    ConfusionMatrix = np.zeros((2, 2))
    total = 0

    curmodel = load_model('animalmodel.h5')

    CatPath = glob.glob("data/validation/cat/*.jpg")
    DogPath = glob.glob("data/validation/dog/*.jpg")
    Predictions = [["Image", "Cat Probability", "Dog Probability", "Actual Class ", "Predicted Class"]]

    observed = -1  # Current observed class
    for ClassPath in [CatPath, DogPath]:
        total += len(ClassPath)  # keeps track of number of validation images
        observed += 1  # first class in alphabetical order is class zero and second is class 1 etc.
        for item in ClassPath:
            pImg = process_image(item, image_targetsize)
            model_prediction = curmodel.predict(pImg)
            predicted = np.argmax(model_prediction)
            ConfusionMatrix[observed, predicted] += 1
            Predictions.append([item, model_prediction[0][0], model_prediction[0][1], observed, predicted])

    Accuracy = 100 * (ConfusionMatrix[0, 0] + ConfusionMatrix[1, 1]) / total

    print("Validation Accuracy: ", Accuracy)
    print("Total validation images: ", total)

    print("Confusion Matrix (Counts):\n", ConfusionMatrix)
    np.savetxt("ConfusionMatrix.csv", ConfusionMatrix, delimiter=",", fmt='%s')

    print("Confusion Matrix (Percentage):\n", np.round_((ConfusionMatrix / total), 4) * 100)
    np.savetxt("ConfusionMatrix_Percentage.csv", np.round_((ConfusionMatrix / total), 4) * 100, delimiter=",", fmt='%s')

    print("Predictions have been save to Predictions.csv")
    np.savetxt("Predictions.csv", Predictions, delimiter=",", fmt='%s')
    return None


def train_model(training_imagesize=224, batchsize=32, epochs=25):
    keras.backend.clear_session()

    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_tensor=Input(shape=(training_imagesize, training_imagesize, 3)))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=preds)

    # Assigning which layers are to be trainined/unfrozen
    # for layer in model.layers[:20]:
    #     layer.trainable = True
    # for layer in model.layers[20:]:
    #     layer.trainable = True


    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       rotation_range=20,
                                       zoom_range=0.15,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.15,
                                       horizontal_flip=True,
                                       vertical_flip=True)

    train_generator = train_datagen.flow_from_directory('./data/train/',
                                                        # this is where you specify the path to the main data folder
                                                        target_size=(training_imagesize, training_imagesize),
                                                        color_mode='rgb',
                                                        batch_size=batchsize,
                                                        class_mode='categorical',
                                                        shuffle=True)

    adam_optimizer = Adam(lr=0.0001, decay=0.9)
    model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])  #

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies
    test_generator = test_datagen.flow_from_directory('./data/test/',

                                                      target_size=(training_imagesize, training_imagesize),
                                                      color_mode='rgb',
                                                      batch_size=batchsize,
                                                      class_mode='categorical',
                                                      shuffle=True)

    step_size_train = train_generator.n // train_generator.batch_size

    #Large model support callback to control the amount of gpu ram being used to prevent out of ram error
    lms_callback = LMSKerasCallback()
    fit_history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=step_size_train,
                                      epochs=epochs,
                                      validation_data=test_generator,
                                      validation_steps=test_generator.n // test_generator.batch_size,
                                      callbacks=[lms_callback])

    # Reporting of the model for analysis purposes
    plt.figure(1, figsize=(15, 8))
    plt.subplot(221)
    plt.plot(fit_history.history['acc'])
    plt.plot(fit_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'])

    plt.subplot(222)
    plt.plot(fit_history.history['loss'])
    plt.plot(fit_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'])
    plt.savefig("Graphical Outputs.pdf")
    plt.clf()  # clears entire figure

    trainloss = fit_history.history['loss']
    testloss = fit_history.history['val_loss']

    trainaccuracy = fit_history.history['acc']
    testaccuracy = fit_history.history['val_acc']

    np.savetxt("train_loss.csv", trainloss, delimiter=",", fmt='%s')
    np.savetxt("test_loss.csv", testloss, delimiter=",", fmt='%s')
    np.savetxt("train_accuracy.csv", trainaccuracy, delimiter=",", fmt='%s')
    np.savetxt("test_accuracy.csv", testaccuracy, delimiter=",", fmt='%s')

    # Saved model which is erased from memory so there is no chance of interference
    model.save('animalmodel.h5')

    return None


def clear_old_training_data():
    import shutil, os
    try:
        shutil.rmtree("./data")
    except:
        print("[WARNING]: At runtime the data folder did not exist. A new data folder is being created.")

    main_dir = ["data"]
    common_dir = ["train", "test", "validation"]
    classes = ["cat", "dog"]
    for dir1 in main_dir:
        for dir2 in common_dir:
            for dir3 in classes:
                try:
                    os.makedirs(os.path.join(dir1, dir2, dir3))
                except OSError:
                    pass


def createtrainingdata(source, destination, validationsplit=0.05, testsplit=0.15, seed=456):
    """
    Creates the training, testing and validation set for the model
    :param source: Path of the original dataset containing folders of each class
    :param destination: Path to destination where data will be moved
    :param validationsplit: Pencentage of data used to validate model after training and testing
    :param testsplit: Percentage of data used for testing the model
    :param seed: Random seed to be used for np.random function
    :return: None
    """
    import shutil
    import numpy as np
    import os

    Classes = ["cat", "dog"]
    try:
        for animalclass in Classes:
            np.random.seed(seed)
            files = os.listdir(source + "/" + animalclass)
            for f in files:
                if np.random.rand(1) < validationsplit:
                    shutil.copy(source + "/" + animalclass + '/' + f,
                                destination + '/validation/' + animalclass + '/' + f)
                elif validationsplit <= np.random.rand(1) < (validationsplit + testsplit):
                    shutil.copy(source + "/" + animalclass + '/' + f, destination + '/test/' + animalclass + '/' + f)
                else:
                    shutil.copy(source + "/" + animalclass + '/' + f, destination + '/train/' + animalclass + '/' + f)
    except:
        print(
            "[ERROR]: A problem has occurred in creating the dataset for training. Please ensure that the data source path is correct.")


if __name__ == '__main__':
    # Create model data for training
    clear_old_training_data()
    createtrainingdata(source="animal_data", destination="data", validationsplit=0.05, testsplit=0.15, seed=456)

    # Model training input parameters
    # Used to control the input size of images
    image_trainingsize = 128
    batch_size = 8
    epochs = 20
    train_model(training_imagesize=image_trainingsize, batchsize=batch_size, epochs=epochs)
    evaluate_model(image_targetsize=image_trainingsize)
