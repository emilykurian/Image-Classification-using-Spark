# Image-Classification-using-Spark
 The aim of the project is to leverage the distributed computing capabilities of Apache Spark to develop a scalable and efficient image classification system.   
 
## PART 1  
The following is performed in Spyder using PySpark:    

I loaded train images of cats and dogs from a directory and created a Pandas DataFrame that stores the filename and class label (0 for cat, 1 for dog) of each image. Then image preprocessing is done by resizing and performing histogram equalization, and extracts features from them. Two types of features are extracted: raw pixel values and histogram of equalized pixel values. The K-Nearest Neighbor (KNN) classifier is trained on both types of features, and the accuracy of the classifier is evaluated using a test set. The accuracy of the two classifiers is printed.  

Test images from the “test1” file are then imported and the above trained KNN model is then used to predict the class labels for the test images. The predicted labels are written to a CSV file with a format of ImageId and Label.  

## PART 2:- Oxford Dataset  
*	The images are in JPG format so they are loaded using OpenCV's cv2.imread() function.For each file, if it is a JPG file, the image is loaded using cv2.imread() and histogram equalization is applied to the image using a function hist_eq(). The resulting histogram equalized image is added to oxford_list and the file name is added to img_names. 
*	Some of the images were corrupted and were failing to import, therefore a if-else loop was used to continue to the next for-loop whenever an image fails to import. 
*	Additionally, the actual label of the image is determined based on the first letter of the file name: if the first letter is uppercase, the label is set to 'cat', otherwise it is set to 'dog'. The actual label is added to actualLabels.
*	The model is then used to predict the labels for the images in oxford_list and the predicted labels are stored in oxpredictedLabels. After predicting the labels, I calculated the accuracy of the predictions by comparing the predicted labels in oxpredictedLabels with the actual labels in actualLabels. The np.mean() function is used to calculate the mean of a boolean array that results from comparing the two arrays.
*	Finally, the predicted labels and actual labels along with the image names are written to a CSV file 'oxford_predicted_labels_final.csv'. The accuracy of the predictions is also printed to the console.
