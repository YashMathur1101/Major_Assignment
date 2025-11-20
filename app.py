# app.py
from flask import Flask, request, render_template_string
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = joblib.load("savedmodel.pth")

HTML_FORM = """
<!doctype html>
<title>Olivetti Face Predict</title>
<h2>Upload grayscale face image (64x64)</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if pred is not none %}
  <h3>Predicted class: {{ pred }}</h3>
{% endif %}
"""

def preprocess_image(file_stream):
    img = Image.open(io.BytesIO(file_stream)).convert("L")
    img = img.resize((64,64))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten().reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    pred = None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            data = file.read()
            x = preprocess_image(data)
            p = model.predict(x)
            pred = int(p[0])
    return render_template_string(HTML_FORM, pred=pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
