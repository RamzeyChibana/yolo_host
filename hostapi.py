from flask import Flask, request,jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best (1).pt') 

transform = transforms.Compose([
    transforms.Resize((640, 640)),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    image = Image.open(file.stream)
    image=np.array(image)
    results=model(image)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]


    labels = labels.tolist()
    cord = cord.tolist()

    return jsonify({'labels': labels, 'coordinates': cord})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')