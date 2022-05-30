import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write as wavwrite
import os
from sklearn.preprocessing import LabelEncoder
#from keras.models import load_model
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from tensorflow import keras

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

loaded_model = keras.models.load_model('my_model')
labels = ["on", "off", "stop", "go", "yes", "no", "up", "down", "left", "right"]
le = LabelEncoder()
y = le.fit_transform(labels)
classes = list(le.classes_)

ALLOWED_EXTENSIONS = {'wav', 'mp3'}
dirname = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = dirname + "/temp/"

# Create Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Upload files function
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'],
                secure_filename(file.filename))
            file.save(filename)
            return redirect(url_for('classify_and_show_results',
                filename=filename))
    return render_template("index.html")

@app.route('/results', methods=['GET'])
def classify_and_show_results():
    filename = request.args['filename']
    rate, audio = wavfile.read(filename)
    prob = loaded_model.predict(audio.reshape(1,16000,1))
    index = np.argmax(prob[0]) #getting output from model
    prediction= classes[index]
    # Delete uploaded file
    os.remove(filename)
    return render_template("results.html", filename=filename, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
