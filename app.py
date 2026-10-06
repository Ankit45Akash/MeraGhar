from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


# ==========================================
# LOAD MERAGHAR MODEL
# ==========================================

model_data = joblib.load("house_price_model.pkl")

model = model_data["model"]

print("MeraGhar model loaded successfully!")


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# PRICE PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get values from website form

        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        city = request.form["city"]

        parking = int(request.form["parking"])
        pool = int(request.form["pool"])
        resale = int(request.form["resale"])


        # Create input data for ML model

        input_data = pd.DataFrame([
            {
                "Area": area,
                "Bedrooms": bedrooms,
                "City": city,
                "Parking": parking,
                "Pool": pool,
                "Resale": resale
            }
        ])


        # ==========================================
        # PREDICT PRICE
        # ==========================================

        predicted_price = model.predict(input_data)[0]

        predicted_price = max(0, predicted_price)


        # ==========================================
        # FORMAT PRICE
        # ==========================================

        if predicted_price >= 100:

            price_display = f"₹{predicted_price / 100:.2f} Crore"

        else:

            price_display = f"₹{predicted_price:.2f} Lakhs"


        # ==========================================
        # PROPERTY SCORE
        # ==========================================

        score = 50

        if bedrooms >= 3:
            score += 10

        if area >= 1200:
            score += 10

        if parking == 1:
            score += 10

        if pool == 1:
            score += 10

        if resale == 0:
            score += 5

        score = min(score, 100)


        # ==========================================
        # PROPERTY INSIGHT
        # ==========================================

        if score >= 85:

            insight = "Excellent property profile"

        elif score >= 70:

            insight = "Good property profile"

        elif score >= 55:

            insight = "Average property profile"

        else:

            insight = "Basic property profile"


        # ==========================================
        # SHOW RESULT PAGE
        # ==========================================

        return render_template(
            "result.html",

            price=price_display,

            raw_price=predicted_price,

            score=score,

            insight=insight,

            city=city,

            bedrooms=bedrooms,

            area=int(area),

            parking=parking,

            pool=pool,

            resale=resale
        )


    except Exception as e:

        return render_template(
            "result.html",
            error=str(e)
        )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )