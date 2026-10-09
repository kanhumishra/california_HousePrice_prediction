from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)

# Load trained model
Obj = joblib.load("california.joblib")

model = Obj["Modell"]
columns = Obj["Columns"]

print("Model columns:")
print(columns)


@app.route("/")
def main():
    return render_template("index.html", columns=columns)


@app.route("/predict", methods=["GET"])
def predict():

    try:
        INPUT = []

        for column in columns:
            value = request.args.get(column)

            if value is None or value == "":
                return jsonify({
                    "success": False,
                    "error": f"Missing value for {column}"
                }), 400

            INPUT.append(float(value))

        # Prediction
        prediction = model.predict([INPUT])

        predicted_value = float(prediction[0])

        return jsonify({
            "success": True,
            "prediction": predicted_value
        })

    except ValueError:
        return jsonify({
            "success": False,
            "error": "Please enter valid numeric values."
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
