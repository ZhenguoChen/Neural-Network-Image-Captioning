import os
import cPickle
from flask import Flask, render_template, request

__author__ = 'Zhenguo'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def Image_Caption(image):
    return "this is the caption"

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

    for file in request.files.getlist("file"):
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)

        predict = Image_Caption(file)

        captions.append((file.filename, predict))

    with open(caption_file, 'wb') as f:
        cPickle.dump(captions, f)

    display = captions[-7:]

    print display

    return render_template("complete.html", image1='../static/img/'+display[0][0], image2='../static/img/'+display[1][0],
                           image3='../static/img/' + display[2][0], image4='../static/img/' + display[3][0],
                           image5='../static/img/' + display[4][0], image6='../static/img/' + display[5][0],
                           image7='../static/img/' + display[6][0],
                           caption1=display[0][1], caption2=display[1][1], caption3=display[2][1],
                           caption4=display[3][1], caption5=display[4][1], caption6=display[5][1], caption7=display[6][1])
    #return render_template("complete.html", caption1='a', caption2='a', caption3='a',
     #                      caption4='a', caption5='a', caption6='a', caption7='a')

if __name__ == '__main__':
    app.run(port=4555, debug=True)