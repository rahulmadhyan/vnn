import numpy as np
import os
import cv2

def load_images_and_create_label(folder, new_pixels):
    
    """
    Loads images from 
    Resize image to 'new_pixels'
    Returns array of images and array of labels
    """
    
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv2.resize(img, (new_pixels, new_pixels))
            images.append(img)
            if filename.find("cat") == -1:
                # dog 
                labels.append(1)
            else:
                # cat
                labels.append(0)
                
    return images, labels

if __name__ == "__main__":
    	# load train images and labels
        images_train_dataset,labels_train_dataset = load_images_and_create_label('/Users/rahulmadhyan/Documents/AI/Neural Networks/Vanilla NN/Data/train', 64)
        images_train_dataset = np.array(images_train_dataset)
        labels_train_dataset = np.array(labels_train_dataset)

		# load test images and labels
        images_test_dataset,labels_test_dataset = load_images_and_create_label('/Users/rahulmadhyan/Documents/AI/Neural Networks/Vanilla NN/Data/test1', 64)
        images_test_dataset = np.array(images_test_dataset)
        labels_test_dataset = np.array(labels_test_dataset)

        size_train = 5000
        images_train = images_train_dataset[:size_train]
        labels_train = labels_train_dataset[:size_train]

        size_test = 5000
        images_test = images_test_dataset[:size_test]
        labels_test = labels_test_dataset[:size_test]
        
        np.savez('/Users/rahulmadhyan/Documents/AI/Neural Networks/Vanilla NN/vnn/image_data_5000', images_train_dataset = images_train, labels_train_dataset = labels_train,
							 images_test_dataset = images_test, labels_test_dataset = labels_test)