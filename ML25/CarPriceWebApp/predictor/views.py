from django.shortcuts import render
import pickle
import numpy as np

# Load model, encoders, and scaler
model = pickle.load(open('model.pkl', 'rb'))
encoders = pickle.load(open('label_encoders.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def home(request):
    return render(request, 'home.html')

def predict(request):
    if request.method == 'POST':
        try:
            inputs = []
            for col in ['name', 'fuel', 'seller_type', 'transmission', 'owner']:
                user_input = request.POST[col]
                encoder = encoders[col]
                inputs.append(
                    encoder.transform([user_input])[0]
                    if user_input in encoder.classes_
                    else encoder.transform(['Unknown'])[0]
                )

            numerical_values = [
                float(request.POST['year']),
                float(request.POST['km_driven']),
                float(request.POST['mileage']),
                float(request.POST['engine']),
                float(request.POST['max_power']),
                int(request.POST['seats']),
            ]
            scaled_values = scaler.transform([numerical_values])
            inputs.extend(scaled_values[0])

            prediction = model.predict([inputs])[0]
            return render(request, 'result.html', {'price': round(prediction, 2)})
        except Exception as e:
            return render(request, 'error.html', {'error': str(e)})