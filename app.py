import os
import cPickle
from flask import Flask, render_template, request
from image_rnn_predict import captioning

__author__ = 'Zhenguo'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# set prediction
model_path = APP_ROOT+'/models/tensorflow'
vgg_path = APP_ROOT+'/data/vgg16-20160129.tfmodel'

### Parameters ###
dim_embed = 256
dim_hidden = 256
dim_in = 4096
batch_size = 1
learning_rate = 0.001
momentum = 0.9
n_epochs = 25

def Image_Caption(image_path):
    return captioning(image_path)

@app.route('/')
def index():
    return render_template("upload.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static/img/')
    caption_file = os.path.join(APP_ROOT, 'static/img/caption.txt')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    if os.path.exists(caption_file):
        f = open(caption_file, 'rb')
        captions = cPickle.load(f)
        f.close()
    else:
        # set initial value
        captions = [('image1.jpg', 'dog!'), ('image2.jpg', 'dog!'),
                    ('image3.jpg', 'dog!'), ('image4.jpg', 'dog!'),
                    ('image5.jpg', 'dog!'), ('image6.jpg', 'dog!'),
                    ('image7.jpg', 'dog!')]

    image_path = []
    for file in request.files.getlist("file"):
        filename = file.filename
        destination = "".join([target, filename])
        print(destination)
        file.save(destination)
        image_path.append(filename)

    predicts = Image_Caption(['static/img/'+img for img in image_path])

    for (image, predict) in zip(image_path, predicts):
        captions.append((image, predict))

    with open(caption_file, 'wb') as f:
        cPickle.dump(captions, f)

    display = captions[-7:]

    print display

    return render_template("complete.html", image1='../static/img/'+display[6][0], image2='../static/img/'+display[5][0],
                           image3='../static/img/' + display[4][0], image4='../static/img/' + display[3][0],
                           image5='../static/img/' + display[2][0], image6='../static/img/' + display[1][0],
                           image7='../static/img/' + display[0][0],
                           caption1=display[6][1], caption2=display[5][1], caption3=display[4][1],
                           caption4=display[3][1], caption5=display[2][1], caption6=display[1][1], caption7=display[0][1])

if __name__ == '__main__':
    app.run(port=4555, debug=True)