#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('7-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('supervised_learning/data/yolo.h5', 'supervised_learning/data/coco_classes.txt', 0.6, 0.5, anchors)
    predictions, image_paths = yolo.predict('supervised_learning/data/yolo')
    for i, name in enumerate(image_paths):
        if "dog.jpg" in name:
            ind = i
            break
    print(image_paths[ind])
    print(predictions[ind])