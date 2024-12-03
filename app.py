from flask import Flask, render_template, request
import numpy as np
import joblib

# Initializing Flask app
app = Flask(__name__)

# Loading the trained model and scaler
model = joblib.load("model/trained_model.pkl")  # Ensure this file exists
scaler = joblib.load("model/scaler.pkl")  # Ensure this file exists

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input data from form
        input_features = [float(request.form[key]) for key in request.form.keys()]
        input_array = np.array(input_features).reshape(1, -1)
        
        # Normalize the input features
        scaled_input = scaler.transform(input_array)

        # Making prediction
        prediction = model.predict(scaled_input)
        
        return render_template(
            "index.html",
            prediction_text=f"The predicted Median Value (MV) is: ${prediction[0]:,.2f}"
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error occurred: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
