from flask import Flask, request, render_template
import os
from model.streetclip_model import predict_location_hierarchical

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    location_info = None
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(filepath)

            prediction = predict_location_hierarchical(filepath)

            if prediction['landmark']:
                location_query = f"{prediction['landmark']}, {prediction['city']}, {prediction['country']}"
            else:
                location_query = f"{prediction['city']}, {prediction['country']}"

            maps_url = f"https://www.google.com/maps/search/?api=1&query={location_query.replace(' ', '+')}"
            prediction['maps_link'] = maps_url
            prediction['filename'] = image.filename
            location_info = prediction

    return render_template('index.html', result=location_info)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ["PORT"]) 
    app.run(host='0.0.0.0', port=port)
