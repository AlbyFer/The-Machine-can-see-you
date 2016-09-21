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
from PIL import Image
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
print("Modules Imported")

#start_time = time.time()
# -------------------------------------------------------------------------------------------------------------- #

print("Loading Images and functions")

dir_names = list(set(glob.glob(os.path.join("competition_data","train", "*"))\
  ).difference(set(glob.glob(os.path.join("competition_data","train","*.*")))))

dir_names = sorted(dir_names)

# Calculate the number of images in the folder
numberofImages = 0
for folder in dir_names:
    for fileNames in os.walk(folder):
        add = len(fileNames[-1])
        numberofImages += add

def getImageMeans(directory_names, pixels):

    """"# Purpose: Calculate category-wise means of images and rescale them.
    # Inputs: directory_names - a list with the location of each category folder.
    #        pixels - number of pixels to which resizing the images.
    # Output: an array with the category-wise means as row vectors."""

    # Defining initial values
    numberofClasses = np.shape(directory_names)[0]
    means_cat = np.zeros((numberofClasses, pixels ** 2), dtype=float)  # Creates an empty array to store the means row-wise
    label = 0  # Encoding the Plankton category
    images_names = []  # Creates a vector of pictures names
    for folder in directory_names:
        for fileNameDir in os.walk(folder):
            n = 0
            arr = np.zeros((pixels, pixels), np.float)
            for fileName in fileNameDir[2]:
                if fileName[-4:] != ".jpg":
                    continue

                nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
                im = imread(nameFileImage, as_grey=True)
                im = resize(im, (pixels, pixels))  # Resizing is done
                im = np.array(im, dtype=np.float)
                images_names.append(fileName)  # Fills a vector of pictures names

                arr = arr + im
                n += 1  # Number of pictures in a folder

            mean = arr / n
            means_cat[label, 0:pixels ** 2] = np.reshape(mean,
                                                         (1, pixels ** 2))  # Matrix of Class means. One for each class.
            label += 1
    return means_cat


im = getImageMeans(dir_names, 100)
ima = np.reshape(im[-1, :], (100, 100))
plt.imshow(ima, cmap='Greys_r')
plt.show()

def getLargestRegion(props, labelmap, imagethres):

    """# Purpose: Identify the largest region in an image.
    # Inputs: props - a list with properties of each region.
    #        labelmap - list of labels of each area in the picture.
    #        imagethres - the thresholded image
    # Output: a list ."""

    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop

def getFeatures(image):

    """# Purpose: Calculate features of a region: axis ratio, hu moments, pixels in convex area and eccentricity.
    # Inputs: image - an image.
    # Output: the features. An array"""

    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image), 0., 1.0)

    # Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4, 4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr * label_list
    label_list = label_list.astype(int)

    # Get regions properties (a list with a different region features: here we use the axis length)
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)

    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    if ((not maxregion is None) and (maxregion.major_axis_length != 0.0)):
        if maxregion is None:
            ratio = 0.0
        else:
            ratio = np.array([[maxregion.minor_axis_length * 1.0 / maxregion.major_axis_length]])
            hu = np.array([maxregion.moments_hu])
          # area = np.array([[maxregion.convex_area]])
            ecc = np.array([[maxregion.eccentricity]])
    res = np.concatenate((ratio, hu, ecc), axis= 1)
    return res


# ----------------------------------------------------------------------------------------------------------- #
# Calculating the mean images for each class.

print("Calculating class means")
means = getImageMeans(dir_names, 25)

# ----------------------------------------------------------------------------------------------------------- #
# Calculates the Y vector of labels and the X matrix of features as the  differences btw images and the classes mean

print("Getting Response and Features matrix")
pix = 25
numberofClasses = np.shape(dir_names)[0]
numFeatures = numberofClasses + 9
X = np.zeros((numberofImages, numFeatures), dtype= float)
y = np.zeros((numberofImages))
feat = np.zeros((numberofImages, 9), dtype= float)
eucl_dist = np.zeros((numberofImages, numberofClasses), dtype= float)
namesClasses = list()
images = np.zeros((numberofImages, pix*pix), dtype= float)
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
            f = getFeatures(image)              # Get the features
            image = resize(image, (pix, pix))  # Resizing is done
            image = np.array(image, dtype=np.float)
            image = np.reshape(image, (1, pix ** 2))
            images[i, 0:pix*pix] = image
            y[i] = label                        # Vector of classes labels
            feat[i, :] = f                 # Vectors of region features
            c = 0
            for t in range(0, np.shape(means)[0]):
                dist = np.linalg.norm(image - means[t, :])  # Euclidean distance
                eucl_dist[i, c] = dist  # Matrix of features
                c += 1
            i += 1
    label += 1
    print("Progress: ", label, " of 121")

X = np.concatenate((eucl_dist, feat), axis= 1)    # Matrix of input variables

# ----------------------------------------------------------------------------------------------------------- #
# Constructing and fitting the Neural Network

print("Start fitting NN")
layer_1 = Layer("Tanh", units= 180)
layer_2 = Layer("Tanh", units = 150)
layer_out = Layer("Softmax")            # Next try, add an hidden layer
lay = [layer_1, layer_2, layer_out]
nn = Classifier(layers= lay, learning_rate= 0.001, n_iter= 15) # Maybe increase the number of iterations
nn.fit(X= X, y= y)
print("Saving Model")

# Saving the model
pickle.dump(nn, open("Big_f_2_layer_15Iterations.pk1", "wb"))

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
    nn_cv = Classifier(layers=lay, learning_rate=0.001, n_iter=15)
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

