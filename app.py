# import numpy as np
# from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from flask import Flask, render_template, request, abort
import pickle  # will help to dump and load ML model

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']
UPLOAD_FOLDER = 'uploads/'

app = Flask(__name__, template_folder='Template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# model = pickle.load(open('model.pkl', 'rb'))

filenames_list = set()


@app.route('/')
def home():
    global filenames_list
    filenames_list = set()
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Upload files
    filenames_list = upload(request.files)

    print(list(filenames_list))

    return render_template('results.html')

    # return list(filenames_list)


def upload(my_files):
    """  Function to upload files  """
    for item in my_files:
        uploaded_file = my_files.get(item)

        # Check allowed file
        if allowed_file(uploaded_file.filename):
            uploaded_file.filename = secure_filename(uploaded_file.filename)

            # Save file
            uploaded_file.save(os.path.join(
                app.config['UPLOAD_FOLDER'], uploaded_file.filename))
            filenames_list.add(uploaded_file.filename)

    return filenames_list


def allowed_file(filename):
    """  Function to check allowed extension  """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    app.run(debug=True)
