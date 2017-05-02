import os
import cPickle
from flask import Flask, render_template, request
from predict_second import *

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

if not os.path.exists('data/ixtoword.npy'):
    print ('You must run 1. O\'reilly Training.ipynb first.')
else:
    with open(vgg_path, 'rb') as f:
        fileContent = f.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)

    images = tf.placeholder("float32", [1, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={"images": images})

    ixtoword = np.load('data/ixtoword.npy').tolist()
    n_words = len(ixtoword)
    maxlen = 15
    graph = tf.get_default_graph()
    sess = tf.InteractiveSession(graph=graph)
    caption_generator = Caption_Generator(dim_in, dim_hidden, dim_embed, batch_size, maxlen + 2, n_words)
    graph = tf.get_default_graph()

    image, generated_words = caption_generator.build_generator(maxlen=maxlen)

    saver = tf.train.Saver()
    print(model_path)
    saved_path = tf.train.latest_checkpoint(model_path)
    print(saved_path)
    saver.restore(sess, saved_path)

print('finish initialization')


def Captioning(sess, image, generated_words, ixtoword, test_image_path=0):  # Naive greedy search

    feat = read_image(test_image_path)
    fc7 = sess.run(graph.get_tensor_by_name("import/Relu_1:0"), feed_dict={images: feat})

    generated_word_index = sess.run(generated_words, feed_dict={image: fc7})
    generated_word_index = np.hstack(generated_word_index)
    generated_words = [ixtoword[x] for x in generated_word_index]
    punctuation = np.argmax(np.array(generated_words) == '.') + 1

    generated_words = generated_words[:punctuation]
    generated_sentence = ' '.join(generated_words)
    print(generated_sentence)
    return generated_sentence

def Image_Caption(image_path):
    predict = Captioning(sess, image, generated_words, ixtoword, image_path)
    return predict

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
        destination = "".join([target, filename])
        print(destination)
        file.save(destination)

        predict = Image_Caption(destination)

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

if __name__ == '__main__':
    app.run(port=4555, debug=True)