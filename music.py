import os
import pickle
from flask import Flask, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import scipy.io.wavfile
from python_speech_features import mfcc
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'wav'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Music/Image Classification</title>
    <h1>Upload New File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # return send_from_directory(app.config['UPLOAD_FOLDER'],
    #                            filename)
    if filename.endswith('wav'):
        filename = UPLOAD_FOLDER + filename
        model = pickle.load(open('finalized_model_lda.sav', 'rb'))
        classification = classify(filename, model)
        return '''<html>
    <header><title>Results</title></header>
    <body>
    This is {0} music!
    </body>
    </html>'''.format(classification)
    else:
        from clarifai.rest import ClarifaiApp

        clarifai_app = ClarifaiApp(api_key='c3f7038ace5d4115a6fe21b5474c4822')

        # get the general model
        model = clarifai_app.models.get("general-v1.3")

        # predict with the model
        result = model.predict_by_filename(UPLOAD_FOLDER + filename)
        lst_result = '<br>'.join([d['name'] + ' ' + str(d['value']) for d in result['outputs'][0]['data']['concepts']])

        return '''<html>
    <header><title>This is title</title></header>
    <body>
    {0}
    </body>
    </html>'''.format(lst_result)



def classify(file, model):
    sample_rate, X = scipy.io.wavfile.read(file)
    mfcc_features = mfcc(X, sample_rate)
    X = mfcc_features.flatten()[:30000]
    prediction = model.predict(X)[0]
    if prediction == 0:
        return 'classical'
    elif prediction == 1:
        return 'metal'
    else:
        return 'country'
    


if __name__ == '__main__':
   app.run()