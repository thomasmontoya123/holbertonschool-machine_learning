0x09. Transfer Learning
=======================

 
Resources
---------

**Read or watch**:

-   [A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning](https://intranet.hbtn.io/rltoken/NiJJLP131eIBOhyzW_8MTA "A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning")
-   [Transfer Learning](https://intranet.hbtn.io/rltoken/4NuXO5rWno8j5WICOJRUmA "Transfer Learning")
-   [Transfer learning & fine-tuning](https://intranet.hbtn.io/rltoken/Y1-2QzyrmmcV1fJWbzk7FA "Transfer learning & fine-tuning")

**Definitions to skim:**

-   [Transfer learning](https://intranet.hbtn.io/rltoken/iDLig1rnDoigSnqiqaxcYg "Transfer learning")

**References:**

-   [Keras Applications](https://intranet.hbtn.io/rltoken/tbgCxEaDctl-CBoEe1hl8g "Keras Applications")
-   [Keras Datasets](https://intranet.hbtn.io/rltoken/Ji49yZq_AF5XkQXrp9CVUg "Keras Datasets")
-   [tf.keras.layers.Lambda](https://intranet.hbtn.io/rltoken/VVxWUZmuV43EajhxyxAbKw "tf.keras.layers.Lambda")
-   [tf.keras.backend.resize_images](https://intranet.hbtn.io/rltoken/snC8wvyvcS2z9K8jWMjb7A "tf.keras.backend.resize_images")
-   [A Survey on Deep Transfer Learning](https://intranet.hbtn.io/rltoken/094hW_tsJrotSljWeiCSSA "A Survey on Deep Transfer Learning")

Learning Objectives
-------------------

At the end of this project, you are expected to be able to [explain to anyone](https://intranet.hbtn.io/rltoken/gzMye98WMCw3bLTsV-ui0Q "explain to anyone"), **without the help of Google**:

### General

-   What is a transfer learning?
-   What is fine-tuning?
-   What is a frozen layer? How and why do you freeze a layer?
-   How to use transfer learning with Keras applications

Requirements
------------

### General

-   Allowed editors: `vi`, `vim`, `emacs`
-   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
-   Your files will be executed with `numpy` (version 1.15) and `tensorflow` (version 1.12)
-   All your files should end with a new line
-   The first line of all your files should be exactly `#!/usr/bin/env python3`
-   A `README.md` file, at the root of the folder of the project, is mandatory
-   Your code should use the `pycodestyle` style (version 2.4)
-   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
-   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
-   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
-   Unless otherwise noted, you are not allowed to import any module except `import tensorflow.keras as K`
-   All your files must be executable
-   The length of your files will be tested using `wc`

* * * * *

Tasks
-----


#### 0\. Transfer Knowledge mandatory

Write a python script that trains a convolutional neural network to classify the CIFAR 10 dataset:

-   You must use one of the applications listed in [Keras Applications](https://intranet.hbtn.io/rltoken/tbgCxEaDctl-CBoEe1hl8g "Keras Applications")
-   Your script must save your trained model in the current working directory as `cifar10.h5`
-   Your saved model should be compiled
-   Your saved model should have a validation accuracy of 87% or higher
-   Your script should not run when the file is imported
-   **Hint1:** *The training and tweaking of hyperparameters may take a while so start early!*
-   **Hint2:** *The CIFAR 10 dataset contains 32x32 pixel images, however most of the Keras applications are trained on much larger images. Your first layer should be a lambda layer that scales up the data to the correct size. I suggest using bilinear interpolation*
-   **Hint3:** *You will want to freeze most of the application layers. Since these layers will always produce the same output, you should compute the output of the frozen layers ONCE and use those values as input to train the remaining trainable layers. This will save you A LOT of time.*

In the same file, write a function `def preprocess_data(X, Y):` that pre-processes the data for your model:

-   `X` is a `numpy.ndarray` of shape `(m, 32, 32, 3)` containing the CIFAR 10 data, where m is the number of data points
-   `Y` is a `numpy.ndarray` of shape `(m,)` containing the CIFAR 10 labels for `X`
-   Returns: `X_p, Y_p`
    -   `X_p` is a `numpy.ndarray` containing the preprocessed `X`
    -   `Y_p` is a `numpy.ndarray` containing the preprocessed `Y`

**NOTE:** *About half of the points for this project are for the blog post in the next task. While you are attempting to train your model, keep track of what you try and why so that you have a log to reference when it is time to write your report.*

```
alexa@ubuntu-xenial:0x09-transfer_learning$ cat 0-main.py
#!/usr/bin/env python3

import tensorflow.keras as K
preprocess_data = __import__('0-transfer').preprocess_data

# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10.h5')
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
alexa@ubuntu-xenial:0x09-transfer_learning$ ./0-main.py
10000/10000 [==============================] - 159s 16ms/sample - loss: 0.3329 - acc: 0.8864

```

**Repo:**

-   GitHub repository: `holbertonschool-machine_learning`
-   Directory: `supervised_learning/0x09-transfer_learning`
-   File: `0-transfer.py`