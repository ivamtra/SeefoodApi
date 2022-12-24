from flask import Flask, request
from flask_cors import CORS, cross_origin
from PIL import Image
import numpy as np
from utils import get_predicted_value


Upload = 'static/upload'
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['uploadFolder'] = Upload

@cross_origin()
@app.route('/')
def index():
  value = {
       "message": "Hello, World!"
    }
  return value



@app.route('/images', methods=['POST'])
def result():
    img = np.array(Image.open(request.files['image']))
    predicted = get_predicted_value(img)
    return_data = {
      "prediction": predicted
    }
    return return_data

if __name__ == '__main__':
  app.run(debug=True, port=8500)
