import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
from flask import render_template
import PIL
from PIL import Image



UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

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
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
	# return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
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

if __name__ == '__main__':
   app.run()