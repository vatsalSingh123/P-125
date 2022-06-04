from flask import Flask,jsonify,request
from classifier import get_prediction

app = Flask(__name__)

@app.route("/pred-digit",methods=['POST'])
def predict_data():
    img = request.files.get("digit")
    prediction = get_prediction(img)
    return jsonify({
    "prediction":prediction
    }),200
    
if __name__ == "__main__":
    app.run(debug=True)