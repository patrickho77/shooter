from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

application = Flask(__name__)

# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- ML Model Code --------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

@application.route('/')
@application.route('/about')
def about():

    return render_template("about.html")

@application.route('/shooterPredictor')
def shooterPredictor():

    return render_template("shooterPredictor.html")

def test_image(path):
    file = open("shooter.pkl", "rb")
    model = joblib.load(file)
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    predicted_prob = prediction[0][1]  # Probability of the positive class (Shooter)
    predicted_class = 1 if predicted_prob >= 0.8 else 0
    class_labels = {0: 'Cellphone', 1: 'Shooter'}
    predicted_label = class_labels[predicted_class]
    plt.imshow(image.array_to_img(img_array[0]))
    plt.title(f"Predicted class: {predicted_label}")
    plt.axis('off')
    plt.show()
    return predicted_label

   
   
@application.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            predicted_label = test_image(file_path)
            return render_template("result.html", label=predicted_label)
    return redirect(url_for('shooterPredictor'))

if __name__ == "__main__":
    application.run(debug=True)