
from flask import Flask, request, redirect, url_for, jsonify, send_from_directory
from time import time
import cv2
import hashlib
# import caffe
# import vqa_data_provider_layer
# from vqa_data_provider_layer import LoadVQADataProvider
import numpy as np
import os
from skimage.transform import resize

import spacy
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
from keras import backend as K
from python_speech_features import mfcc
K.set_image_data_format('channels_first')

import seaborn
import librosa
import scipy.io.wavfile
import pickle


# constants
# GPU_ID = 3
# RESNET_MEAN_PATH = "../00_data_preprocess/ResNet_mean.binaryproto"
# RESNET_LARGE_PROTOTXT_PATH = "../00_data_preprocess/ResNet-152-448-deploy.prototxt"
# RESNET_CAFFEMODEL_PATH = "/x/daylen/ResNet-152-model.caffemodel"
# EXTRACT_LAYER = "res5c"
# EXTRACT_LAYER_SIZE = (2048, 14, 14)
# TARGET_IMG_SIZE = 448
# VQA_PROTOTXT_PATH = "/x/daylen/saved_models/multi_att_2_glove/proto_test_batchsize1.prototxt"
# VQA_CAFFEMODEL_PATH = "/x/daylen/saved_models/multi_att_2_glove/_iter_190000.caffemodel"
# VDICT_PATH = "/x/daylen/saved_models/multi_att_2_glove/vdict.json"
# ADICT_PATH = "/x/daylen/saved_models/multi_att_2_glove/adict.json"
VQA_weights_file_name = 'models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = 'models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name = 'models/CNN/vgg16_weights.h5'

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG'])
MUSIC_EXTENSIONS = set(['wav'])
UPLOAD_FOLDER = './uploads/'
# VIZ_FOLDER = './viz/'

# global variables
app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# resnet_mean = None
# resnet_net = None
# vqa_net = None
feature_cache = {}
image_features = None
last_file_type = None
# vqa_data_provider = LoadVQADataProvider(VDICT_PATH, ADICT_PATH, batchsize=1, \
# mode='test', data_shape=EXTRACT_LAYER_SIZE)

# helpers


def setup():
    # global resnet_mean
    # global resnet_net
    # global vqa_net
    # # data provider
    # vqa_data_provider_layer.CURRENT_DATA_SHAPE = EXTRACT_LAYER_SIZE

    # # mean substraction
    # blob = caffe.proto.caffe_pb2.BlobProto()
    # data = open( RESNET_MEAN_PATH , 'rb').read()
    # blob.ParseFromString(data)
    # resnet_mean = np.array( caffe.io.blobproto_to_array(blob)).astype(np.float32).reshape(3,224,224)
    # resnet_mean = np.transpose(cv2.resize(np.transpose(resnet_mean,(1,2,0)), (448,448)),(2,0,1))

    # # resnet
    # caffe.set_device(GPU_ID)
    # caffe.set_mode_gpu()

    # # resnet_net = caffe.Net(RESNET_LARGE_PROTOTXT_PATH, RESNET_CAFFEMODEL_PATH, caffe.TEST)

    # # # our net
    # # vqa_net = caffe.Net(VQA_PROTOTXT_PATH, VQA_CAFFEMODEL_PATH, caffe.TEST)

    # uploads
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # if not os.path.exists(VIZ_FOLDER):
    #     os.makedirs(VIZ_FOLDER)

    print('Finished setup')


def trim_image(img):
    y, x, c = img.shape
    if c != 3:
        raise Exception('Expected 3 channels in the image')
    resized_img = cv2.resize(img, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    transposed_img = np.transpose(resized_img, (2, 0, 1)).astype(np.float32)
    ivec = transposed_img - resnet_mean
    return ivec


def make_rev_adict(adict):
    """
    An adict maps text answers to neuron indices. A reverse adict maps neuron
    indices to text answers.
    """
    rev_adict = {}
    for k, v in adict.items():
        rev_adict[v] = k
    return rev_adict


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def allowed_music_format(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in MUSIC_EXTENSIONS


def softmax(arr):
    e = np.exp(arr)
    dist = e / np.sum(e)
    return dist


def downsample_image(img):
    img_h, img_w, img_c = img.shape
    img = resize(img, (448 * img_h / img_w, 448))
    return img


def get_image_model(CNN_weights_file_name):
    ''' Takes the CNN weights file, and returns the VGG model update
    with the weights. Requires the file VGG.py inside models/CNN '''
    from models.CNN.VGG import VGG_16
    image_model = VGG_16(CNN_weights_file_name)

    # this is standard VGG 16 without the last two layers
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # one may experiment with "adam" optimizer, but the loss function for
    # this kind of task is pretty standard
    image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return image_model


def get_image_features(image_file_name, CNN_weights_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the
    weights (filters) as a 1, 4096 dimension vector '''
    image_features = np.zeros((1, 4096))
    # Magic_Number = 4096  > Comes from last layer of VGG Model

    # Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))

    # The mean pixel values are taken from the VGG authors, which are the values computed from the training dataset.
    mean_pixel = [103.939, 116.779, 123.68]

    im = im.astype(np.float32, copy=False)
    for c in range(3):
        im[:, :, c] = im[:, :, c] - mean_pixel[c]

    im = im.transpose((2, 0, 1))  # convert the image to RGBA

    # this axis dimension is required becuase VGG was trained on a dimension
    # of 1, 3, 224, 224 (first axis is for the batch size
    # even though we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0)

    image_features[0, :] = get_image_model(
        CNN_weights_file_name).predict(im)[0]
    return image_features


def get_VQA_model(VQA_weights_file_name):
    ''' Given the VQA model and its weights, compiles and returns the model '''

    from models.VQA.VQA import VQA_MODEL
    vqa_model = VQA_MODEL()
    vqa_model.load_weights(VQA_weights_file_name)

    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model


def get_question_features(question):
    ''' For a given question, a unicode string, returns the timeseris vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
    word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
        question_tensor[0, j, :] = tokens[j].vector
    return question_tensor

# def save_attention_visualization(source_img_path, att_map, dest_name):
#     """
#     Visualize the attention map on the image and save the visualization.
#     """
#     img = cv2.imread(source_img_path) # cv2.imread does auto-rotate

#     # downsample source image
#     img = downsample_image(img)
#     img_h, img_w, img_c = img.shape

#     _, att_h, att_w = att_map.shape
#     att_map = att_map.reshape((att_h, att_w))

#     # upsample attention map to match original image
#     upsample0 = resize(att_map, (img_h, img_w), order=3) # bicubic interpolation
#     upsample0 = upsample0 / upsample0.max()

#     # create rgb-alpha
#     rgba0 = np.zeros((img_h, img_w, img_c + 1))
#     rgba0[..., 0:img_c] = img
#     rgba0[..., 3] = upsample0

#     path0 = os.path.join(VIZ_FOLDER, dest_name + '.png')
#     cv2.imwrite(path0, rgba0 * 255.0)

#     return path0

# routes


@app.route('/', methods=['GET'])
def index():
    print('here')
    return app.send_static_file('index.html')


@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    global image_features, last_file_type
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file was uploaded.'})
    if allowed_file(file.filename):
        start = time()
        last_file_type = 'image'
        file_hash = hashlib.md5(file.read()).hexdigest()
        if file_hash in feature_cache:
            json = {'img_id': file_hash, 'time': time() - start}
            return jsonify(json)
        save_path = os.path.join(
            app.config['UPLOAD_FOLDER'], file_hash + '.jpg')
        file.seek(0)
        file.save(save_path)
        # if img is None:
        #     return jsonify({'error': 'Error reading image.'})
        image_features = get_image_features(save_path, CNN_weights_file_name)
        feature_cache[file_hash] = image_features
        json = {'img_id': file_hash, 'time': time() - start}
        return jsonify(json)
    elif allowed_music_format(file.filename):
        start = time()
        last_file_type = 'music'
        file_hash = hashlib.md5(file.read()).hexdigest()
        if file_hash in feature_cache:
            json = json = {'img_id': file_hash, 'time': time() - start}
            return jsonify(json)
        save_path = os.path.join(
            app.config['UPLOAD_FOLDER'], file_hash + '.wav')
        file.seek(0)
        file.save(save_path)

        y, sr = librosa.load(save_path)
        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        feature_cache[file_hash] = {'tempo': tempo}
        json = {'img_id': file_hash, 'time': time() - start}
        return jsonify(json)
    else:
        return jsonify({'error': 'Please upload a JPG or PNG.'})


@app.route('/api/upload_question', methods=['POST'])
def upload_question():
    if last_file_type == 'image':
        img_hash = request.form['img_id']
        if img_hash not in feature_cache:
            return jsonify({'error': 'Unknown image ID. Try uploading the image again.'})
        start = time()
        img_feature = feature_cache[img_hash]
        question = request.form['question']
        # img_ques_hash = hashlib.md5(img_hash + question).hexdigest()
        question_features = get_question_features(question)
        vqa_model = get_VQA_model(VQA_weights_file_name)
        y_output = vqa_model.predict([question_features, image_features])
        y_sort_index = np.argsort(y_output)

        top_answers = []
        top_scores = []
        labelencoder = joblib.load(label_encoder_file_name)
        for label in reversed(y_sort_index[0, -5:]):
            output = str(round(
                y_output[0, label] * 100, 2)).zfill(5), "% ", labelencoder.inverse_transform(label)
            top_answers.append(output[2])
            top_scores.append(float(output[0]) / 100)

        json = {'answer': top_answers[0],
                'answers': top_answers,
                'scores': top_scores,
                'time': time() - start}
        print(json)
        return jsonify(json)
    elif last_file_type == 'music':
        music_hash = request.form['img_id']
        if music_hash not in feature_cache:
            return jsonify({'error': 'Unknown music ID. Try uploading the music file again.'})
        start = time()
        music_feature = feature_cache[music_hash]

        question = request.form['question']
        if 'tempo' in question:
            tempo = music_feature['tempo'][0]
            print(type(tempo))
            json = {'answer': 'Tempo',
                    'answers': ['Tempo'],
                    'scores': [tempo],
                    'time': time() - start}
            print(json)
            return jsonify(json)
        elif 'genre' in question:
            n_mfcc = 12
            directories = ['classical', 'country', 'disco', 'metal', 'pop']
            model = pickle.load(open('finalized_model_svc_5.sav', 'rb'))
            scaler = pickle.load(open('svc_scaler_5.sav', 'rb'))
            save_path = os.path.join(
                app.config['UPLOAD_FOLDER'], music_hash + '.wav')
            X, sample_rate = librosa.load(save_path)
            mfcc_features = librosa.feature.mfcc(
                X, sr=sample_rate, n_mfcc=n_mfcc).T
            mfcc_scaled = scaler.transform(mfcc_features)
            predicted_labels = model.predict(
                mfcc_scaled[int(len(mfcc_scaled) * .1):int(len(mfcc_scaled) * .9)])
            prediction = np.argmax([(predicted_labels == c).sum()
                                    for c in range(len(directories))])
            prediction = directories[prediction]

            json = {'answer': 'Genre',
                    'answers': ['Genre'],
                    'scores': [prediction],
                    'time': time() - start}
            print(json)
            return jsonify(json)

        else:
            return jsonify({'error': 'Unknown Question. Try asking a different question.'})
    else:
        return jsonify({'error': 'Unknown file. Try uploading the file again.'})
# @app.route('/viz/<filename>')
# def get_visualization(filename):
#     return send_from_directory(VIZ_FOLDER, filename)


if __name__ == '__main__':
    setup()
    app.run()
