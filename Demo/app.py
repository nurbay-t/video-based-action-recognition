from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18
from torchvision.io import read_video

# Initialize Flask app
app = Flask(__name__)

classes = ['Diving-Side', 'Golf-Swing', 'Kicking', 'Lifting', 'Riding-Horse', 
           'SkateBoarding-Front', 'Swing-Bench', 'Swing-SideAngle', 
           'boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']


# Load model
device = torch.device("cpu")
model = r2plus1d_18()
num_classes = 14
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
model.eval()


# Define preprocess function
def process_video(video_path, frame_count=16, resize=(224, 224)):
    video, _, _ = read_video(video_path)
    video = video.float()

    # Resize and pick `frame_count` frames
    processed_video = []
    for i in range(min(frame_count, video.shape[0])):
        frame = video[i]
        if frame.size(2) == 1:
            frame = frame.repeat(1, 1, 3)
        frame = frame.permute(2, 0, 1)
        frame = F.interpolate(frame.unsqueeze(0), size=resize, mode='bilinear', align_corners=False).squeeze(0)
        processed_video.append(frame)

    while len(processed_video) < frame_count:
        processed_video.append(processed_video[-1])

    processed_video = torch.stack(processed_video, dim=0).permute(1, 0, 2, 3)
    return processed_video

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    video_path = 'temp_video.mp4'
    file.save(video_path)

    video_tensor = process_video(video_path)  # Ensure you have this function defined
    with torch.no_grad():
        outputs = model(video_tensor.unsqueeze(0))
        probabilities = F.softmax(outputs, dim=1)
        top5_prob, top5_pred = torch.topk(probabilities, 5)

        # Prepare the prediction results
        predictions = [{'class': classes[pred], 'probability': prob.item() * 100}
                       for pred, prob in zip(top5_pred[0], top5_prob[0])]

    return jsonify({'result': predictions})

if __name__ == '__main__':
    app.run(debug=True)
