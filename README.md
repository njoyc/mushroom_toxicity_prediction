# 🍄 Mushroom Toxicity Prediction

A machine learning web app that predicts whether a mushroom is **edible or poisonous** based on its physical characteristics. Built using **Flask** and **Scikit-learn**, the app leverages a **Random Forest** classifier trained on the popular UCI Mushroom dataset, achieving over **99% accuracy**.

## 🔍 About the Project

Mushroom foraging can be dangerous if toxic species are misidentified. This project helps users assess mushroom edibility by providing a simple interface for prediction using trained machine learning models.

### 🧠 Features
- Predicts mushroom edibility (edible/poisonous) based on user input.
- Interactive web UI built with Flask.
- Uses a Random Forest model trained on categorical mushroom traits.
- Deployed for real-time usage.
- High prediction accuracy (~99%).

## 🚀 Demo
 
📂 [GitHub Repository](https://github.com/njoyc/mushroom_toxicity_prediction)

## ⚙️ Tech Stack

- **Python**
- **Flask**
- **Scikit-learn**
- **HTML/CSS**
- **Jupyter Notebook**

## 📊 Dataset

- **Source**: [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)
- **Records**: 8,124 mushrooms
- **Features**: 22 categorical features (e.g., cap shape, color, odor)

## 🏗️ How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/njoyc/mushroom_toxicity_prediction.git
   cd mushroom_toxicity_prediction
````

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask app:

   ```bash
   python app.py
   ```

5. Open in browser: `http://localhost:5000`

## 📈 Model Performance

* **Model**: Random Forest Classifier
* **Accuracy**: 99.3%
* **Training Time**: \~1 second

## 📌 Screenshots

![App Screenshot](screenshots/predict_form.png)
*Prediction form UI*

![Prediction Result](screenshots/result.png)
*Prediction result displayed after submission*

## 🙌 Acknowledgments

* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom)
* Scikit-learn documentation
* Flask web framework

## 📬 Contact

**Author**: [@njoyc](https://github.com/njoyc)
📧 Email: [your.email@example.com](mailto:your.email@example.com)

---

> ⚠️ Disclaimer: This tool is for educational purposes only. Do not consume wild mushrooms based solely on its predictions.

```


