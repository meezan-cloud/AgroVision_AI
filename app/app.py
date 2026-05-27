# Importing essential libraries and modules
from flask import Flask, render_template, request, jsonify
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
import requests
import config   # API KEY FILE

# ==============================================================================================

# ------------------------- LOADING MODELS -----------------------------------------------

disease_classes = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
    'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

crop_recommendation_model = pickle.load(open('models/RandomForest.pkl', 'rb'))

# ==============================================================================================
# 🔥 WEATHER FUNCTION

def weather_fetch(city_name):
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    try:
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url, timeout=5)
        data = response.json()

        if data["cod"] != 200:
            raise Exception("City not found")

        temperature = round((data["main"]["temp"] - 273.15), 2)
        humidity = data["main"]["humidity"]

        return temperature, humidity

    except:
        return 25, 60   # fallback

# ==============================================================================================

def predict_image(img, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()]

# ==============================================================================================

app = Flask(__name__)

# HOME
@app.route('/')
def home():
    return render_template('index.html', title='AgroSmart AI - Home')

# CROP PAGE
@app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html', title='AgroSmart AI - Crop Recommendation')

# FERTILIZER PAGE
@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer.html', title='AgroSmart AI - Fertilizer')

# ==============================================================================================
# 🔥 WEATHER API ROUTE (AUTO-FILL)

@app.route('/get-weather', methods=['POST'])
def get_weather():
    try:
        data = request.get_json()
        city = data.get("city")

        temperature, humidity = weather_fetch(city)

        return jsonify({
            "temperature": temperature,
            "humidity": humidity
        })

    except:
        return jsonify({
            "temperature": 25,
            "humidity": 60
        })

# ==============================================================================================
# CROP PREDICTION

@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    try:
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        temperature, humidity = weather_fetch(city)

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        prediction = crop_recommendation_model.predict(data)[0]

        return render_template(
            'crop-result.html',
            prediction=prediction,
            title='AgroSmart AI - Result'
        )

    except:
        return render_template('try_again.html')

# ==============================================================================================
# FERTILIZER

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K

    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]

    if max_value == "N":
        key = 'NHigh' if n < 0 else "Nlow"
    elif max_value == "P":
        key = 'PHigh' if p < 0 else "Plow"
    else:
        key = 'KHigh' if k < 0 else "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response)

# ==============================================================================================
# DISEASE

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    if request.method == 'POST':
        file = request.files.get('file')

        if not file:
            return render_template('disease.html')

        try:
            img = file.read()
            prediction = predict_image(img)
            prediction = Markup(str(disease_dic[prediction]))

            return render_template('disease-result.html', prediction=prediction)

        except:
            return render_template('disease.html')

    return render_template('disease.html')

# ==============================================================================================

if __name__ == '__main__':
    app.run(debug=True)