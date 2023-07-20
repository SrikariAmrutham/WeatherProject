from flask import Flask, request, Response, jsonify
from keras.models import load_model
from keras.utils import load_img, img_to_array 
import numpy as np
import tensorflow
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
model = load_model('model.h5')
model.make_predict_function()
dic = {0:'Cloudy', 1:'Rain', 2:'Sunny', 3:'Sunrise'}

# funtion to predict the image
def predict_label(img_path):
  i = load_img(img_path , target_size = (250 , 250))
  i = img_to_array(i, dtype=np.uint8)
  i = np.array(i)/255.0
  predict = model.predict(i[np.newaxis , ...])
  predicted_class = dic[np.argmax(predict[0] , axis = -1)]
  return predicted_class

@app.route('/', methods=['POST'])
def index():
    image_file = request.files['image']
    img_path = "static/" + image_file.filename	
    image_file.save(img_path)
    p = predict_label(img_path)
    return jsonify({'predicted_class': p})

if __name__ == "_main_":
    app.run(debug=True)