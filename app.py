import keras
# using Image from Piloow library of python to resize and convert images into desired format and size
import Image
from flask import Flask
from flask import render_template, request
import os
import numpy as np
from keras.preprocessing import image
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


app = Flask(__name__)

UPLOAD_FOLDER = "static"

@app.route("/", methods = ["GET", "POST"])
def upload():
    if request.method == "POST":
        imagefile = request.files['image']
        if imagefile:
            im = Image.open(imagefile)
            im = im.resize((96, 96))
            im.save(r'static\\test.tif')

            image_location = os.path.join(UPLOAD_FOLDER, 'test.tif')


            with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
                new_model = keras.models.load_model("CNN_KaggleModel_18-0.94.hdf5")

            test_image = keras.preprocessing.image.load_img(image_location)
            test_image = image.img_to_array(test_image)
            test_image = test_image / 255.0
            test_image = np.expand_dims(test_image, axis=0)
            prediction = new_model.predict(test_image)

            p1 = round(prediction[0][0] * 100, 2)
            p2 = round(prediction[0][1] * 100, 2)
            result = [[p1, p2]]


            return render_template("index.html", prediction=result)

    return render_template("index.html", prediction = [[0, 0]])

# @app.route("/pred")


if __name__ == '__main__':
    app.run(debug=True)
