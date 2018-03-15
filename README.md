# Images Captioning Using Neural Network -- Keras

###### [YouTube](https://www.youtube.com/watch?v=f2waevH1b6I) 
###### [Blog](https://zhenguochen.github.io/image-captioning-using-neural-network-cnn-lstm/)

### Challenges!
##### Baseline:
* KNN model for image captioning. Get the features of images using SURF or GIST algorithm, and feed into knn model.
* For prediction, find the closest image based on features.
* use BLEU score to choose one best caption from the captions of closest images.
##### Final Model:
* Use VGG16 or VGG19 CNN to extraction features from images.
* Use LSTM model to generate the captions

### Dataset:
* [Flickr8K](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)
* [Flickr30k](https://drive.google.com/file/d/0B5o40yxdA9PqTnJuWGVkcFlqcG8/view)
* [captions](https://drive.google.com/file/d/0B2vTU3h54lTydXFjSVM5T2t4WmM/view)

## run baseline
* run image_knn.py

## training LSTM model
* run image_rnn.py

## LSTM prediction
* run image_rnn_predict.py

## VGG Feature Extractor (CNN)
* use 16 layer version of CNN to extract features
* [pre-trained model](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) put pr-trained model file in model folder

## Present Results
* to present results, we use Flask and show the result as web pages.
* run python app.py, and use 127.0.0.1:4555 to see the results in browser.

## dependencies:
* OpenCV: for feature extraction(SIFT, SURF, ORB)
run install-opencv.sh
* GIST: A wrapper for Lear's GIST implementation written in C.
follow the instruction: [here](https://github.com/yuichiroTCY/lear-gist-python)
* Tensorflow
* Keras
* Flask
