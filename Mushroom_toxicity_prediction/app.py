from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset (ensure the path is correct)
df = pd.read_csv("C:/Users/joyne/Downloads/mushroom_cleaned2.csv")
print(df.shape)

# Train the model
X = df.drop(columns=['class'])  # Ensure 'target_column' is correctly referenced
y = df['class']  # Ensure it corresponds to your target column
model = RandomForestClassifier()
model.fit(X, y)


# Example route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction (POST request)
@app.route('/predict', methods=['POST'])
def predict():
    # Collect the form data (user input)
    try:
        cap_diameter = float(request.form['cap-diameter'])
        cap_shape = int(request.form['cap-shape'])
        gill_attachment = int(request.form['gill-attachment'])
        gill_color = int(request.form['gill-color'])
        stem_height = float(request.form['stem-height'])
        stem_width = float(request.form['stem-width'])
        stem_color = int(request.form['stem-color'])
        season = float(request.form['season'])
    except KeyError as e:
        return jsonify({"error": f"Missing input field: {e}"}), 400

    # Prepare the input features in the same order as the model was trained
    input_features = [cap_diameter, cap_shape, gill_attachment, gill_color, stem_height, stem_width, stem_color, season]
    
    # Convert input into a DataFrame (same format as training data)
    input_data = pd.DataFrame([input_features], columns=df.columns[:-1])  # Exclude 'target_column' from features
    
    # Predict toxicity (0 = Non-Toxic, 1 = Toxic)
    prediction = model.predict(input_data)
    
    # Return the result to the frontend
    result = "Toxic" if prediction[0] == 1 else "Non-Toxic"
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
