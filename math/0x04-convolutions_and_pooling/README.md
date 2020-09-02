0x04. Convolutions and Pooling
==============================

Specializations - Machine Learning ― Math

_by Alexa Orrico, Software Engineer at Holberton School_


![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/11/ed9ca14839ad0201f19e.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123755Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2f32162b52bc669871e9bbde8f7b524e53f353b88d75a6e7e1d1217f136b0cef)

Resources
---------

**Read or watch**:

*   [Convolution](/rltoken/xbzvTRaBX2LUOM7A1NazVQ "Convolution")
*   [Kernel (image processing)](/rltoken/lsI2xbijDWAiKDFuCYkcAA "Kernel (image processing)")
*   [Image Kernels](/rltoken/Qeq8i5dhkR9Tlp-IgFDzQw "Image Kernels")
*   [Undrestanding Convolutional Layers](/rltoken/g8kHsJFzC51whRSEupvidw "Undrestanding Convolutional Layers")
*   [What is max pooling in convolutional neural networks?](/rltoken/crEEAb4sDHc30ntPwY-qsQ "What is max pooling in convolutional neural networks?")
*   [Edge Detection Examples](/rltoken/nV4RcnhzFvjLfl7z2k5-Cw "Edge Detection Examples") (_Note: I suggest watching this video at 1.5x - 2x speed_)
*   [Padding](/rltoken/WZ_a9ntwdJ_AU51W46KOlw "Padding") (_Note: I suggest watching this video at 1.5x - 2x speed_)
*   [Strided Convolutions](/rltoken/yupMT890fCjD5XVyogDkmg "Strided Convolutions") (_Note: I suggest watching this video at 1.5x - 2x speed_)
*   [Convolutions over Volumes](/rltoken/vdFQg1m-0BJ_s0lg8b3fkg "Convolutions over Volumes") (_Note: I suggest watching this video at 1.5x - 2x speed_)
*   [Pooling Layers](/rltoken/Z0dPond1Oi9a04MiWsbgXA "Pooling Layers") (_Note: I suggest watching this video at 1.5x - 2x speed_)
*   [numpy.pad](/rltoken/QkWjIyjvPImhaA4HJGGz-w "numpy.pad")
*   [A guide to convolution arithmetic for deep learning](/rltoken/ZJItcZYPPp4e6bAV-xaMkw "A guide to convolution arithmetic for deep learning")

Learning Objectives
-------------------


*   What is a convolution?
*   What is max pooling? average pooling?
*   What is a kernel/filter?
*   What is padding?
*   What is “same” padding? “valid” padding?
*   What is a stride?
*   What are channels?
*   How to perform a convolution over an image
*   How to perform max/average pooling over an image

Requirements
------------

### General

*   Allowed editors: `vi`, `vim`, `emacs`
*   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
*   Your files will be executed with `numpy` (version 1.15)
*   All your files should end with a new line
*   The first line of all your files should be exactly `#!/usr/bin/env python3`
*   A `README.md` file, at the root of the folder of the project, is mandatory
*   Your code should use the `pycodestyle` style (version 2.5)
*   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
*   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
*   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
*   Unless otherwise noted, you are not allowed to import any module except `import numpy as np`
*   You are not allowed to use `np.convolve`
*   All your files must be executable
*   The length of your files will be tested using `wc`

More Info
---------

### Testing

Please download [this dataset](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-ml/animals_1.npz "this dataset") for use in some of the following main files.

* * *

Tasks
-----


#### 0\. Valid Convolution mandatory

Write a function `def convolve_grayscale_valid(images, kernel):` that performs a valid convolution on grayscale images:

*   `images` is a `numpy.ndarray` with shape `(m, h, w)` containing multiple grayscale images
    *   `m` is the number of images
    *   `h` is the height in pixels of the images
    *   `w` is the width in pixels of the images
*   `kernel` is a `numpy.ndarray` with shape `(kh, kw)` containing the kernel for the convolution
    *   `kh` is the height of the kernel
    *   `kw` is the width of the kernel
*   You are only allowed to use two `for` loops; any other loops of any kind are not allowed
*   Returns: a `numpy.ndarray` containing the convolved images
```
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 0-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/MNIST.npz')
        images = dataset['X_train']
        print(images.shape)
        kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
        images_conv = convolve_grayscale_valid(images, kernel)
        print(images_conv.shape)
    
        plt.imshow(images[0], cmap='gray')
        plt.show()
        plt.imshow(images_conv[0], cmap='gray')
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./0-main.py 
    (50000, 28, 28)
    (50000, 26, 26)
```    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123755Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=3743a2b5f0b4cfa06f84c05bd3b2fd216bee4ff7af52d3699f1a02b1d59b36a7)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6e1b02cc87497f12f17e.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123755Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=11fd90b2e730d7c3a7c2abed76ac8da88ccce0599294cc1bff8b62c2070bb039)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x04-convolutions_and_pooling`
*   File: `0-convolve_grayscale_valid.py`


#### 1\. Same Convolution mandatory

Write a function `def convolve_grayscale_same(images, kernel):` that performs a same convolution on grayscale images:

*   `images` is a `numpy.ndarray` with shape `(m, h, w)` containing multiple grayscale images
    *   `m` is the number of images
    *   `h` is the height in pixels of the images
    *   `w` is the width in pixels of the images
*   `kernel` is a `numpy.ndarray` with shape `(kh, kw)` containing the kernel for the convolution
    *   `kh` is the height of the kernel
    *   `kw` is the width of the kernel
*   if necessary, the image should be padded with 0’s
*   You are only allowed to use two `for` loops; any other loops of any kind are not allowed
*   Returns: a `numpy.ndarray` containing the convolved images
```
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 1-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_grayscale_same = __import__('1-convolve_grayscale_same').convolve_grayscale_same
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/MNIST.npz')
        images = dataset['X_train']
        print(images.shape)
        kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
        images_conv = convolve_grayscale_same(images, kernel)
        print(images_conv.shape)
    
        plt.imshow(images[0], cmap='gray')
        plt.show()
        plt.imshow(images_conv[0], cmap='gray')
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./1-main.py 
    (50000, 28, 28)
    (50000, 28, 28)
 ```   

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123755Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=3743a2b5f0b4cfa06f84c05bd3b2fd216bee4ff7af52d3699f1a02b1d59b36a7)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/b32bba8fea86011c3372.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123755Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=73627791474a6733ee5bf165903ece824a1615f82694bc6c8294c17f187642ef)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x04-convolutions_and_pooling`
*   File: `1-convolve_grayscale_same.py`


#### 2\. Convolution with Padding mandatory

Write a function `def convolve_grayscale_padding(images, kernel, padding):` that performs a convolution on grayscale images with custom padding:

*   `images` is a `numpy.ndarray` with shape `(m, h, w)` containing multiple grayscale images
    *   `m` is the number of images
    *   `h` is the height in pixels of the images
    *   `w` is the width in pixels of the images
*   `kernel` is a `numpy.ndarray` with shape `(kh, kw)` containing the kernel for the convolution
    *   `kh` is the height of the kernel
    *   `kw` is the width of the kernel
*   `padding` is a tuple of `(ph, pw)`
    *   `ph` is the padding for the height of the image
    *   `pw` is the padding for the width of the image
    *   the image should be padded with 0’s
*   You are only allowed to use two `for` loops; any other loops of any kind are not allowed
*   Returns: a `numpy.ndarray` containing the convolved images
```
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 2-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_grayscale_padding = __import__('2-convolve_grayscale_padding').convolve_grayscale_padding
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/MNIST.npz')
        images = dataset['X_train']
        print(images.shape)
        kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
        images_conv = convolve_grayscale_padding(images, kernel, (2, 4))
        print(images_conv.shape)
    
        plt.imshow(images[0], cmap='gray')
        plt.show()
        plt.imshow(images_conv[0], cmap='gray')
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./2-main.py 
    (50000, 28, 28)
    (50000, 30, 34)
 ```   

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123755Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=3743a2b5f0b4cfa06f84c05bd3b2fd216bee4ff7af52d3699f1a02b1d59b36a7)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/3f178b675c1e2fdc86bd.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123755Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2bce37e1b434de6b42ee03ddbd625230b33c7c3a0c61b7bd84e53902975d9917)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x04-convolutions_and_pooling`
*   File: `2-convolve_grayscale_padding.py`


#### 3\. Strided Convolution mandatory

Write a function `def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):` that performs a convolution on grayscale images:

*   `images` is a `numpy.ndarray` with shape `(m, h, w)` containing multiple grayscale images
    *   `m` is the number of images
    *   `h` is the height in pixels of the images
    *   `w` is the width in pixels of the images
*   `kernel` is a `numpy.ndarray` with shape `(kh, kw)` containing the kernel for the convolution
    *   `kh` is the height of the kernel
    *   `kw` is the width of the kernel
*   `padding` is either a tuple of `(ph, pw)`, ‘same’, or ‘valid’
    *   if ‘same’, performs a same convolution
    *   if ‘valid’, performs a valid convolution
    *   if a tuple:
        *   `ph` is the padding for the height of the image
        *   `pw` is the padding for the width of the image
    *   the image should be padded with 0’s
*   `stride` is a tuple of `(sh, sw)`
    *   `sh` is the stride for the height of the image
    *   `sw` is the stride for the width of the image
*   You are only allowed to use two `for` loops; any other loops of any kind are not allowed _Hint: loop over `i` and `j`_
*   Returns: a `numpy.ndarray` containing the convolved images
```
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 3-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/MNIST.npz')
        images = dataset['X_train']
        print(images.shape)
        kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
        images_conv = convolve_grayscale(images, kernel, padding='valid', stride=(2, 2))
        print(images_conv.shape)
    
        plt.imshow(images[0], cmap='gray')
        plt.show()
        plt.imshow(images_conv[0], cmap='gray')
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./3-main.py 
    (50000, 28, 28)
    (50000, 13, 13)
```    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/17e3fb852b947ff6d845.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123755Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=3743a2b5f0b4cfa06f84c05bd3b2fd216bee4ff7af52d3699f1a02b1d59b36a7)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/036ccba7dccf211dab76.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123755Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=05520d0c5555c3b927a6bf7818f3289ad8cc98464282c27afecff8916a302959)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x04-convolutions_and_pooling`
*   File: `3-convolve_grayscale.py`


#### 4\. Convolution with Channels mandatory

Write a function `def convolve_channels(images, kernel, padding='same', stride=(1, 1)):` that performs a convolution on images with channels:

*   `images` is a `numpy.ndarray` with shape `(m, h, w, c)` containing multiple images
    *   `m` is the number of images
    *   `h` is the height in pixels of the images
    *   `w` is the width in pixels of the images
    *   `c` is the number of channels in the image
*   `kernel` is a `numpy.ndarray` with shape `(kh, kw, c)` containing the kernel for the convolution
    *   `kh` is the height of the kernel
    *   `kw` is the width of the kernel
*   `padding` is either a tuple of `(ph, pw)`, ‘same’, or ‘valid’
    *   if ‘same’, performs a same convolution
    *   if ‘valid’, performs a valid convolution
    *   if a tuple:
        *   `ph` is the padding for the height of the image
        *   `pw` is the padding for the width of the image
    *   the image should be padded with 0’s
*   `stride` is a tuple of `(sh, sw)`
    *   `sh` is the stride for the height of the image
    *   `sw` is the stride for the width of the image
*   You are only allowed to use two `for` loops; any other loops of any kind are not allowed
*   Returns: a `numpy.ndarray` containing the convolved images
```
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 4-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve_channels = __import__('4-convolve_channels').convolve_channels
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/animals_1.npz')
        images = dataset['data']
        print(images.shape)
        kernel = np.array([[[0, 0, 0], [-1, -1, -1], [0, 0, 0]], [[-1, -1, -1], [5, 5, 5], [-1, -1, -1]], [[0, 0, 0], [-1, -1, -1], [0, 0, 0]]])
        images_conv = convolve_channels(images, kernel, padding='valid')
        print(images_conv.shape)
    
        plt.imshow(images[0])
        plt.show()
        plt.imshow(images_conv[0])
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./4-main.py 
    (10000, 32, 32, 3)
    (10000, 30, 30)
 ```   

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123755Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=1cd037623a947b8732b0ecc56c2ba742bbddf3945f21f68c03aeff34808597ee)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/8bc039fb38d60601b01a.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123755Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d35a6f828c33f7e62ed115a6db2733b22d488e012514ad337827c06c19e8000e)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x04-convolutions_and_pooling`
*   File: `4-convolve_channels.py`


#### 5\. Multiple Kernels mandatory

Write a function `def convolve(images, kernels, padding='same', stride=(1, 1)):` that performs a convolution on images using multiple kernels:

*   `images` is a `numpy.ndarray` with shape `(m, h, w, c)` containing multiple images
    *   `m` is the number of images
    *   `h` is the height in pixels of the images
    *   `w` is the width in pixels of the images
    *   `c` is the number of channels in the image
*   `kernels` is a `numpy.ndarray` with shape `(kh, kw, c, nc)` containing the kernels for the convolution
    *   `kh` is the height of a kernel
    *   `kw` is the width of a kernel
    *   `nc` is the number of kernels
*   `padding` is either a tuple of `(ph, pw)`, ‘same’, or ‘valid’
    *   if ‘same’, performs a same convolution
    *   if ‘valid’, performs a valid convolution
    *   if a tuple:
        *   `ph` is the padding for the height of the image
        *   `pw` is the padding for the width of the image
    *   the image should be padded with 0’s
*   `stride` is a tuple of `(sh, sw)`
    *   `sh` is the stride for the height of the image
    *   `sw` is the stride for the width of the image
*   You are only allowed to use three `for` loops; any other loops of any kind are not allowed
*   Returns: a `numpy.ndarray` containing the convolved images
```
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 5-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    convolve = __import__('5-convolve').convolve
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/animals_1.npz')
        images = dataset['data']
        print(images.shape)
        kernels = np.array([[[[0, 1, 1], [0, 1, 1], [0, 1, 1]], [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], [[0, -1, 1], [0, -1, 1], [0, -1, 1]]],
                           [[[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]], [[5, 0, 0], [5, 0, 0], [5, 0, 0]], [[-1, -1, 0], [-1, -1, 0], [-1, -1, 0]]],
                           [[[0, 1, -1], [0, 1, -1], [0, 1, -1]], [[-1, 0, -1], [-1, 0, -1], [-1, 0, -1]], [[0, -1, -1], [0, -1, -1], [0, -1, -1]]]])
    
        images_conv = convolve(images, kernels, padding='valid')
        print(images_conv.shape)
    
        plt.imshow(images[0])
        plt.show()
        plt.imshow(images_conv[0, :, :, 0])
        plt.show()
        plt.imshow(images_conv[0, :, :, 1])
        plt.show()
        plt.imshow(images_conv[0, :, :, 2])
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./5-main.py 
    (10000, 32, 32, 3)
    (10000, 30, 30, 3)
 ```   

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123755Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=1cd037623a947b8732b0ecc56c2ba742bbddf3945f21f68c03aeff34808597ee)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6d6319bb470e3566e885.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123755Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=af5dfb20a02d4e990be0dbe8749c08606ca6c3222142ab39a6ff76f498898c89)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/1370dd6200e942eee8f9.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123756Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=904b9e92cfdab73a61dfb7d367de06a3b38ee20d10ba3a2a4fe4eeb206711028)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/a24b7d741b3c378f9f89.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123756Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=18249528e7907eb7a248229f38c480b5c564543cf8a0600348e8c8e1b7e536ec)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x04-convolutions_and_pooling`
*   File: `5-convolve.py`


#### 6\. Pooling mandatory

Write a function `def pool(images, kernel_shape, stride, mode='max'):` that performs pooling on images:

*   `images` is a `numpy.ndarray` with shape `(m, h, w, c)` containing multiple images
    *   `m` is the number of images
    *   `h` is the height in pixels of the images
    *   `w` is the width in pixels of the images
    *   `c` is the number of channels in the image
*   `kernel_shape` is a tuple of `(kh, kw)` containing the kernel shape for the pooling
    *   `kh` is the height of the kernel
    *   `kw` is the width of the kernel
*   `stride` is a tuple of `(sh, sw)`
    *   `sh` is the stride for the height of the image
    *   `sw` is the stride for the width of the image
*   `mode` indicates the type of pooling
    *   `max` indicates max pooling
    *   `avg` indicates average pooling
*   You are only allowed to use two `for` loops; any other loops of any kind are not allowed
*   Returns: a `numpy.ndarray` containing the pooled images
```
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ cat 6-main.py 
    #!/usr/bin/env python3
    
    import matplotlib.pyplot as plt
    import numpy as np
    pool = __import__('6-pool').pool
    
    
    if __name__ == '__main__':
    
        dataset = np.load('../../supervised_learning/data/animals_1.npz')
        images = dataset['data']
        print(images.shape)
        images_pool = pool(images, (2, 2), (2, 2), mode='avg')
        print(images_pool.shape)
    
        plt.imshow(images[0])
        plt.show()
        plt.imshow(images_pool[0] / 255)
        plt.show()
    ubuntu@alexa-ml:~/math/0x04-convolutions_and_pooling$ ./6-main.py 
    (10000, 32, 32, 3)
    (10000, 16, 16, 3)
```    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/6add724c812e8dcddb21.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123756Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=65357f1babec43697eccc6ee5d444603d062d489f21dd49611b3419ca268e388)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/12/ab4705f939c3a8e487bb.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200603%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200603T123756Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=67b84d9cbd93f3f8af6626b0d005e2c1deaf8dd147d9684e0cac1e680ce104ca)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x04-convolutions_and_pooling`
*   File: `6-pool.py`