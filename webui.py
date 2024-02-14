from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Загрузка модели
model = tf.keras.models.load_model('mnist_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Получаем изображение от пользователя
        img_file = request.files['image'].read()
        img = Image.open(io.BytesIO(img_file)).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img).reshape((1, 28, 28, 1)) / 255.0

        # Делаем предсказание
        predictions = model.predict(img_array)
        digit = np.argmax(predictions)

        return jsonify({'digit': int(digit)})

if __name__ == '__main__':
    app.run(debug=True)

