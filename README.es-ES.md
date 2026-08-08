

# Generación de Leyendas para Imágenes Usando Red Neuronal -- Keras

###### [YouTube](https://www.youtube.com/watch?v=f2waevH1b6I) 
###### [Blog](https://zhenguochen.github.io/image-captioning-using-neural-network-cnn-lstm/)

### ¡Desafíos!
##### Línea base:
* Modelo KNN para la generación de leyendas de imágenes. Obtener las características de las imágenes mediante los algoritmos SURF o GIST, y alimentar el modelo KNN.
* Para la predicción, buscar la imagen más cercana en función de las características.
* Usar la puntuación BLEU para elegir la mejor leyenda entre las de las imágenes más cercanas.
##### Modelo Final:
* Usar VGG16 o VGG19 CNN para extraer características de las imágenes.
* Usar un modelo LSTM para generar las leyendas

### Conjunto de datos:
* [Flickr8K](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)
* [Flickr30k](https://drive.google.com/file/d/0B5o40yxdA9PqTnJuWGVkcFlqcG8/view)
* [captions](https://drive.google.com/file/d/0B2vTU3h54lTydXFjSVM5T2t4WmM/view)

## Ejecutar línea base
* ejecutar image_knn.py

## Entrenar modelo LSTM
* ejecutar image_rnn.py

## Predicción con LSTM
* ejecutar image_rnn_predict.py

## Extractor de Características VGG (CNN)
* usar la versión de 16 capas de CNN para extraer características
* [modelo pre-entrenado](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz) colocar el archivo del modelo pre-entrenado en la carpeta model

## Mostrar Resultados
* Para presentar los resultados, utilizamos Flask y mostrar los resultados como páginas web.
* ejecutar python app.py, y usar 127.0.0.1:4555 para ver los resultados en el navegador.

## Dependencias:
* OpenCV: para la extracción de características (SIFT, SURF, ORB)
ejecutar install-opencv.sh
* GIST: Un adaptador para la implementación GIST de Lear escrita en C.
seguir las instrucciones: [aquí](https://github.com/yuichiroTCY/lear-gist-python)
* Tensorflow
* Keras
* Flask
