import pandas as pd
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from flask_cors import CORS

def preprocess(df):
    df["Title"] = df["Title"].replace({
        "Miss": "Ms",
        "Ms": "Ms"
    })
    df["Title"] = df["Title"].replace({
        "Rev": "Rare",
        "Dr": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Don": "Rare",
        "Lady": "Rare",
        "Sir": "Rare",
        "Mlle": "Rare",
        "Jonkheer": "Rare"
    })  
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df = df.drop(columns=["SibSp"])
    df = df.drop(columns=["Parch"])

    # One-hot encode
    df = pd.get_dummies(df, columns=["Sex", "Embarked", "Title"])

    expected_cols = [
        'Pclass', 'Age', 'Fare', 'FamilySize',
        'Sex_female', 'Sex_male',
        'Embarked_C', 'Embarked_Q', 'Embarked_S',
        'Title_Master', 'Title_Mr', 'Title_Mrs', 'Title_Ms', 'Title_Rare'
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]

    return df

app = Flask(__name__)
CORS(app)

# load model terbaik
model = joblib.load("best_model.pkl")

# Home route
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Ensure input is in a DataFrame
    input_df = pd.DataFrame([data])
    
    # Preprocess input
    processed_df = preprocess(input_df)
    
    # Predict using the model
    prediction = model.predict(processed_df)[0]
    
    return jsonify({"survived": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)