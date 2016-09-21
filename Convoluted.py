print("Importing libraries")
import time
from skimage.io import imread
from skimage.transform import resize
from sknn.mlp import Classifier, Convolution, Layer
import glob
import os
import pickle
import numpy as np
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report, log_loss
import warnings
from skimage import morphology
from skimage import measure
warnings.filterwarnings("ignore")
print("Modules Imported")

start_time = time.time()
# -------------------------------------------------------------------------------------------------------------- #

print("Loading and preparing features datasets")

dir_names = list(set(glob.glob(os.path.join("competition_data","train", "*"))\
  ).difference(set(glob.glob(os.path.join("competition_data","train","*.*")))))

dir_names = sorted(dir_names)

# Calculate the number of images in the folder
numberofImages = 0
for folder in dir_names:
    for fileNames in os.walk(folder):
        add = len(fileNames[-1])
        numberofImages += add

    # ----------------------------------------------------------------------------------------------------------- #
    # Calculates the Y vector of labels and the X matrix of features as the  differences btw images and the classes mean

pix = 25
X = np.zeros((numberofImages, pix**2), dtype=float)
y = np.zeros((numberofImages))
namesClasses = list()
label = 0
i = 0

for folder in dir_names:
    currentClass = folder.split(os.pathsep)[-1]     # Creates a list of classes names as strings
    namesClasses.append(currentClass)               # Idem
    for fileNameDir in os.walk(folder):
        for fileName in fileNameDir[2]:
            if fileName[-4:] != ".jpg":
                continue
            nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
            image = imread(nameFileImage, as_grey=True)
            image = resize(image, (pix, pix))  # Resizing is done
            image = np.array(image, dtype=np.float)
            image = np.reshape(image, (1, pix ** 2))
            X[i, 0:pix * pix] = image
            y[i] = label
            i += 1
    label += 1
    print("Progress: ", label, " of 121")

# Fitting a convoluted neural network
print("Start fitting convoluted NN")
layer_1 = Convolution("Tanh", channels= 6, kernel_shape= (1,3))
layer_out = Layer("Softmax")
lay = [layer_1, layer_out]
nn = Classifier(layers= lay, learning_rate= 0.001, n_iter= 2)

print("Start fitting NN")
nn.fit(X= X, y= y)
print("Fineshed fitting")

# Saving the NN
pickle.dump(nn, open("Convoluted.pk1", "wb"))

# ----------------------------------------------------------------------------------------------------------- #
# Estimating the generalisation error with CV: all classes indivudually and multiclass log-loss

print("CV for class-wise generalisation errors")
num_folds = 2
kf = KFold(y, n_folds=num_folds)
y_pred = y * 0
l_loss = np.zeros((num_folds,1), dtype= float)
p = 0

for train, test in kf:
    X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
    nn_cv = Classifier(layers=lay, learning_rate=0.001, n_iter=2)
    nn_cv.fit(X=X_train, y=y_train)
    y_pred[test] = nn_cv.predict(X_test)
    y_pred2 = nn_cv.predict_proba(X_test)
    l_loss[p, 0] = log_loss(y_test, y_pred2)
    p += 1
print(classification_report(y, y_pred, target_names=namesClasses))
log_loss_CV = np.average(l_loss, axis=0)

# Calculating the multiclass log-loss
print("Multiclass Log-loss by CV: ", log_loss_CV)

print("Finished program")
print("--- %s seconds ---" % (round(time.time() - start_time, 4))) # Calculates machine time for the program
