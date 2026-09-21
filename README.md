````markdown
# 🍄 Mushroom Toxicity Prediction

A machine learning web application that predicts whether a mushroom is **Toxic** or **Non-Toxic** based on its physical characteristics.

The application uses a **Random Forest Classifier** trained on a mushroom dataset and provides a Flask-based web interface for making predictions.

---

## 📌 Project Overview

The goal of this project is to build a machine learning application that can classify mushrooms based on their characteristics.

The user provides information about the mushroom through a web form, including:

- Cap diameter
- Cap shape
- Gill attachment
- Gill color
- Stem height
- Stem width
- Stem color
- Season

The Flask application converts these inputs into numerical features and passes them to the Random Forest model. The model then predicts whether the mushroom belongs to the Toxic or Non-Toxic class.

---

## 🚀 Features

- 🌐 Flask-based web interface
- 🤖 Random Forest classification model
- 🍄 Mushroom toxicity prediction
- 📊 Eight input features
- 🔄 Real-time prediction through the web form
- 🎨 Simple user interface
- ✅ Toxic / Non-Toxic prediction result

---

## 🛠️ Technologies Used

- **Python**
- **Flask**
- **Pandas**
- **Scikit-learn**
- **Random Forest**
- **HTML**
- **CSS**
- **Jupyter Notebook**

---

## 🧠 Machine Learning Model

The project uses a **Random Forest Classifier** for binary classification.

The dataset contains eight input features and one target column:

| Feature | Description |
|---|---|
| Cap Diameter | Diameter of the mushroom cap |
| Cap Shape | Shape of the mushroom cap |
| Gill Attachment | Type of gill attachment |
| Gill Color | Color of the gills |
| Stem Height | Height of the mushroom stem |
| Stem Width | Width of the mushroom stem |
| Stem Color | Color of the stem |
| Season | Season associated with the observation |
| Class | Target variable |

The target class is represented numerically:

```text
0 → Non-Toxic
1 → Toxic
````

---

## 🖥️ Application Screenshots

### Home Page

![Mushroom Toxicity Prediction - Home Page](static/images/home-page.png)

### Example Prediction Input

![Mushroom Toxicity Prediction - Input](static/images/details.png)

### Toxic Prediction

![Mushroom Toxicity Prediction - Toxic Result](static/images/toxic.png)

### Non-Toxic Prediction

![Mushroom Toxicity Prediction - Non-Toxic Result](static/images/non-toxic.png)

---

## 🔄 Application Workflow

```text
User enters mushroom characteristics
              ↓
       Flask web application
              ↓
     Input values are encoded
              ↓
       Random Forest Model
              ↓
        Model prediction
              ↓
     ┌────────┴────────┐
     ↓                 ↓
  Toxic            Non-Toxic
```

---

## 📂 Project Structure

```text
mushroom_toxicity_prediction/
│
├── Mushroom_toxicity_prediction/
│   │
│   ├── static/
│   │   └── images/
│   │
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   │
│   ├── Mushroom_classification2.ipynb
│   ├── app.py
│   └── mushroom_cleaned2.csv
│
├── images/
│   ├── home-page.png
│   ├── details.png
│   ├── toxic.png
│   └── non-toxic.png
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/njoyc/mushroom_toxicity_prediction.git
```

### 2. Navigate to the project directory

```bash
cd mushroom_toxicity_prediction
cd Mushroom_toxicity_prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install Flask pandas scikit-learn
```

### 6. Run the application

```bash
python app.py
```

### 7. Open the application

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 🔮 Making a Prediction

1. Open the application in your browser.
2. Enter the mushroom's cap diameter.
3. Select the cap shape.
4. Select the gill attachment.
5. Select the gill color.
6. Enter the stem height.
7. Enter the stem width.
8. Select the stem color.
9. Select the season.
10. Click **Predict**.
11. The application displays either **Toxic** or **Non-Toxic**.

---

## 📓 Machine Learning Notebook

The project includes the Jupyter Notebook:

```text
Mushroom_classification2.ipynb
```

The notebook contains the machine learning work associated with the project, while `app.py` provides the Flask web interface used to make predictions.

---

## 📄 Dataset

The dataset used by the Flask application is:

```text
mushroom_cleaned2.csv
```

The CSV contains the numerical feature values used by the Random Forest classifier along with the target `class` column.

---

## 🔗 Repository

GitHub:
[https://github.com/njoyc/mushroom_toxicity_prediction](https://github.com/njoyc/mushroom_toxicity_prediction)

---

## 👩‍💻 Author

**Joy Christiana Nelapati**

GitHub:
[https://github.com/njoyc](https://github.com/njoyc)

```
And yes: **the four screenshots you selected are all included in this version.**
```
