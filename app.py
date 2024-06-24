# load packages==============================================================
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('Models/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler if it exists
try:
    with open('Models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    scaler = None

# Dictionaries for mapping
Operating_system_dict = {'Android': 0, 'FireTv OS 6': 1, 'WebOS': 3, 'Tizen': 5, 'Google TV': 2, 'Linux': 4}
Speaker_dict = {'20 W Speaker Output': 0, '40 W Speaker Output': 1, '24 W Speaker Output': 2, '30 W Speaker Output': 3,
                '16 W Speaker Output': 4, '50 W Speaker Output': 5, '60 W Speaker Output': 6, '100 W Speaker Output': 7}
Frequency_dict = {'60 Hz Refresh Rate': 0, '50 Hz Refresh Rate': 1, '120 Hz Refresh Rate': 2, '100 Hz Refresh Rate': 3,
                  '200 Hz Refresh Rate': 4}
Picture_qualtiy_dict = {'HD Ready': 0, 'Ultra HD': 1, 'Full HD': 2}
class_names = ['Adsun', 'Croma', 'SAMSUNG', 'LG', 'SONY', 'MOTOROLA', 'Nokia', 'TCL', 'Vu', 'KODAK', 'Haier', 'PHILIPS', 'Thomson', 'iFFALCON', 'Hyundai', 'Lloyd']

def Recommendations(Stars, MRP, Operating_system, Speaker, Frequency, Picture_qualtiy):
    # Map the input values using the dictionaries
    os_mapped = Operating_system_dict.get(Operating_system, None)
    speaker_mapped = Speaker_dict.get(Speaker, None)
    freq_mapped = Frequency_dict.get(Frequency, None)
    pq_mapped = Picture_qualtiy_dict.get(Picture_qualtiy, None)
    
    # Ensure all mappings are valid
    if None in [os_mapped, speaker_mapped, freq_mapped, pq_mapped]:
        raise ValueError("One or more input values could not be mapped.")
    
    # Create a feature vector (example, adjust as needed for your model)
    input_features = np.array([[Stars, MRP, os_mapped, speaker_mapped, freq_mapped, pq_mapped]])
    
    # Apply feature scaling if scaler is available
    if scaler:
        input_features = scaler.transform(input_features)
    
    # Use the model to predict probabilities
    probabilities = model.predict_proba(input_features)[0]
    
    # Combine class names and probabilities
    results = list(zip(class_names, probabilities))
    
    # Sort results by probability in descending order and limit to top 5
    results = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    
    return results

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/pred', methods=['POST'])
def pred():
    if request.method == 'POST':
        Operating_system = request.form['Operating_system1']
        Stars = request.form['stars']
        MRP = int(request.form['MRP'])
        Speaker = request.form['Speaker1']
        Frequency = request.form['Frequency1']
        Picture_qualtiy = request.form['Picture_qualtiy1']
        
        try:
            recommendations = Recommendations(Stars, MRP, Operating_system, Speaker, Frequency, Picture_qualtiy)
        except ValueError as e:
            return f"Input error: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"

        return render_template('result.html', recommendations=recommendations)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
