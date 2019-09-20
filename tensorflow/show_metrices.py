# from sklearn.metrics import accuracy_score  # works
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# # import for showing the confusion matrix
# import itertools
# import operator
# from PIL import Image
# from PIL import ImageDraw
# import numpy as np

def show_classification_matrix(Y_pred, test_labels):
    from sklearn.metrics import accuracy_score  # works
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    import itertools
    import operator
    from PIL import Image
    from PIL import ImageDraw

    from keras.datasets import cifar10
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    # This function will be used to display confusion matrix
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(test_labels, axis=1)
    # # compute the confusion matrix
    print("Confusion matrix:\n%s" % confusion_matrix(y_true=Y_true, y_pred=Y_pred_classes))
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    return confusion_mtx


def show_misclassified_images_cifar10(Y_pred, Y_true_nor, X_test):
    from keras.datasets import cifar10
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    # print(score)
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(Y_true_nor, axis=1)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    # Errors are difference between predicted labels and true labels
    errors = (Y_pred_classes - Y_true != 0)
    errors
    Y_pred_classes_errors = Y_pred_classes[errors]
    pred_errors = Y_pred_classes_errors
    Y_pred_errors = Y_pred[errors]
    Y_true_errors = Y_true[errors]
    obs_errors = Y_true_errors
    X_test_errors = X_test[errors]
    img_errors = X_test_errors
    # Probabilities of the wrong predicted numbers
    Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)
    # Predicted probabilities of the true values in the error set
    true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
    # Difference between the probability of the predicted label and the true label
    delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
    # Sorted list of the delta prob errors
    sorted_dela_errors = np.argsort(delta_pred_true_errors)
    most_important_errors = sorted_dela_errors[-30:]
    errors_index = most_important_errors
    """ This function shows 6 images with their predicted and real labels"""
    #   n = 0
    #   nrows = 2
    #   ncols = 3
    n = 0
    nrows = 6
    ncols = 5

    fig, ax = plt.subplots(nrows, ncols, figsize=(22, 22), sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            #             print('\n')
            #       ax[row,col].imshow((img_errors[error]).reshape((32,32)))
            ax[row, col].imshow((img_errors[error]))
            ax[row, col].set_title(
                "Pred :{}\nTrue :{}".format(class_names[pred_errors[error]], class_names[obs_errors[error]]))
            #             print('\n')
            n += 1
    #             print("value:",n)
    # If you don't do tight_layout() you'll have weird overlaps
    plt.tight_layout()

def cifar10_misclassified_gradcam_heatmap_images(Y_pred,Y_true_nor,X_test,model2):
  from keras.datasets import cifar10
  import time
  import matplotlib.pyplot as plt
  import numpy as np
  import os

  class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

  # print(score)
  # Convert validation observations to one hot vectors
  Y_true = np.argmax(Y_true_nor,axis = 1)
  # Convert predictions classes to one hot vectors
  Y_pred_classes = np.argmax(Y_pred,axis = 1)
  # Errors are difference between predicted labels and true labels
  errors = (Y_pred_classes - Y_true != 0)
  # errors
  Y_pred_classes_errors = Y_pred_classes[errors]
  pred_errors = Y_pred_classes_errors
  Y_pred_errors = Y_pred[errors]
  Y_true_errors = Y_true[errors]
  obs_errors = Y_true_errors

  X_test_errors = X_test[errors]
  img_errors = X_test_errors
  # Probabilities of the wrong predicted numbers
  Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)
  # Predicted probabilities of the true values in the error set
  true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
  # Difference between the probability of the predicted label and the true label
  delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors
  # Sorted list of the delta prob errors
  sorted_dela_errors = np.argsort(delta_pred_true_errors)
  most_important_errors = sorted_dela_errors[-30:]
  errors_index = most_important_errors
  """ This function shows 6 images with their predicted and real labels"""
#   n = 0
#   nrows = 2
#   ncols = 3
  n = 0
  nrows = 6
  ncols = 5

  fig, ax = plt.subplots(nrows,ncols,figsize=(22, 22), sharex=True,sharey=True)
  for row in range(nrows):
    for col in range(ncols):
      error = errors_index[n]
#             print('\n')
#       ax[row,col].imshow((img_errors[error]).reshape((32,32)))
## added for gradcam
      img = np.copy(img_errors[error])
      x = np.expand_dims(img_errors[error], axis=0)
      class_idx = pred_errors[error]
      class_output =  model2.output[:, class_idx] ##modelCiphar.output[:, class_idx
      last_conv_layer = model2.get_layer("conv2d_6")
      grads = K.gradients(class_output, last_conv_layer.output)[0]
      pooled_grads = K.mean(grads, axis=(0, 1, 2))
      iterate = K.function([model2.input], [pooled_grads, last_conv_layer.output[0]])
      pooled_grads_value, conv_layer_output_value = iterate([x])
      for i in range(10):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

      heatmap = np.mean(conv_layer_output_value, axis=-1)
      heatmap = np.maximum(heatmap, 0)
      heatmap /= np.max(heatmap)
      heatmap = cv2.resize(heatmap, (img_errors[error].shape[1], img_errors[error].shape[0]))
      heatmap = np.uint8(255 * heatmap)

      heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
      heatmap = heatmap/255

      for i in range(len(heatmap)):
        for j in range(len(heatmap[0])):
          if heatmap[i][j][1]<=0.01 and heatmap[i][j][2]<=0.01:
            heatmap[i][j] = 0

#       result = Image.blend(img_errors[error], heatmap, alpha=0.5)
      superimposed_img = 0.5*(img_errors[error]) + 0.5*heatmap
      ax[row,col].imshow((superimposed_img))
  ## end of addition for gradcam
#       ax[row,col].imshow((img_errors[error]))
      ax[row,col].set_title("Pred :{}\nTrue :{}".format(class_names[pred_errors[error]],class_names[obs_errors[error]]))
#             print('\n')
      n += 1
#             print("value:",n)
  # If you don't do tight_layout() you'll have weird overlaps
  plt.tight_layout()

def draw_sample(n, rows=4, cols=4, imfile=None, fontsize=12):
  from keras.datasets import cifar10
  import time
  import matplotlib.pyplot as plt
  import numpy as np
  import os

    ## https://samyzaf.com/ML/cifar10/cifar10.html
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = y_train.reshape(y_train.shape[0])  # somehow y_train comes as a 2D nx1 matrix
    ## To test the second utility, let's draw the first 15 images in a 3x5 grid:
    ## draw_sample(0, 3, 5)


    for i in range(0, rows*cols):
        plt.subplot(rows, cols, i+1)
        im = X_train[n+i].reshape(32,32,3)
        plt.imshow(im, cmap='gnuplot2')
        plt.title("{}".format(class_name[y_train[n+i]]), fontsize=fontsize)
        plt.axis('off')
        plt.subplots_adjust(wspace=0.6, hspace=0.01)
        #plt.subplots_adjust(hspace=0.45, wspace=0.45)
        #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if imfile:
        plt.savefig(imfile)

def display_each_class_one_cifar10():
  from keras.datasets import cifar10
  import time
  import matplotlib.pyplot as plt
  import numpy as np
  import os


  num_classes =10
  (train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
  class_names = ['airplane','automobile','bird','cat','deer',
                 'dog','frog','horse','ship','truck']
  fig = plt.figure(figsize=(12,12))
  for i in range(num_classes):
      ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
      idx = np.where(train_labels[:]==i)[0]
      features_idx = train_features[idx,::]
      img_num = np.random.randint(features_idx.shape[0])
      im = features_idx[img_num]
      ax.set_title(class_names[i])
      plt.imshow(im)
  plt.show()

def show_one_class_images_cifar10(classname ='CAT',numimage=10):
  from keras.datasets import cifar10
  import time
  import matplotlib.pyplot as plt
  import numpy as np
  import os

  num_classes = 10
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()

  myclass = np.char.lower(classname)  ## lets convert to lower case to ensure we dont clash
  class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
  # Get the index of elements with value our class
  result = np.where(class_names == myclass)[0][0]
  fig = plt.figure(figsize=(12,12))
  for i in range(numimage):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(y_train[:]==result)[0]
    features_idx = x_train[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = features_idx[img_num]
    ax.set_title(class_names[result])
    plt.imshow(im)
  plt.show()

def draw_image_cifar10(i):
    ## This function is to draw nth image from anuy
    import matplotlib.pyplot as plt
    from keras.datasets import cifar10
    class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
    ##Let's draw image 7 in X_train for example
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    ## draw_img(7)
    im = x_train[i]
    c = y_train[i]
    plt.imshow(im)
    plt.title("Class %d (%s)" % (c, class_name[c]))
    plt.axis('on')


