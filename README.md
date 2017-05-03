# csci5622 project -- Images Captioning

### Challenges!
##### Baseline:
* KNN model for image captioning
##### TARGET:
* CNN and RNN model for image captioning

### Dataset:
* [Flickr8K](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)

## run baseline
* run format.py
* run image_knn.py

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
* Lasagne: install latest version pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
* to run image_rnn_predict.py installation of python library pynlpl is needed. use command, pip install pynlpl(or sudo install pynlpl)
