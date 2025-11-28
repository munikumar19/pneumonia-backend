from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from werkzeug.utils import secure_filename
import csv
import jwt

# App setup
app = Flask(__name__)
CORS(app)

# JWT secret
JWT_SECRET = 'your-secret-key'  # Replace with secure key

# Static route
@app.route('/static/uploads/<filename>')
def serve_file(filename):
    return send_from_directory('static/uploads', filename)

# Auth blueprint
from auth import auth_bp
app.register_blueprint(auth_bp)

# MongoDB setup
client = MongoClient("mongodb+srv://pneumouser:pneumopass123@cluster0.jaxxykj.mongodb.net/?retryWrites=true&w=majority")
db = client["pneumoniaApp"]
collection = db["predictions"]

# Load trained model
# model = tf.keras.models.load_model('model/pneumonia_model.h5')
model = tf.keras.models.load_model("model_converted", compile=False)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check auth token
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user_email = ''
        if token:
            try:
                decoded = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
                user_email = decoded.get('email', '')
            except Exception as e:
                print("❌ Invalid JWT:", e)

        # Validate file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        filename = secure_filename(file.filename)
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({'error': 'Invalid file type. Please upload .png or .jpg only'}), 400

        # Save image
        filepath = os.path.join('static/uploads', filename)
        file.save(filepath)

        # Load and preprocess image dynamically based on model input shape
        model_input_shape = model.input_shape[1:3]  # Typically (150, 150)
        img = image.load_img(filepath, target_size=model_input_shape)
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_tensor)
        print("✅ Raw prediction output:", predictions)

        # Handle binary or categorical
        if predictions.shape[1] == 1:
            # Binary classification (sigmoid output)
            probability = predictions[0][0]
            confidence = float(probability * 100) if probability > 0.5 else float((1 - probability) * 100)
            predicted_class = "Pneumonia Detected" if probability > 0.5 else "Normal"
        else:
            # Categorical (softmax)
            class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions)) * 100
            predicted_class = "Pneumonia Detected" if class_idx == 1 else "Normal"

        # Save to DB
        collection.insert_one({
            "filename": filename,
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "timestamp": datetime.now().isoformat(),
            "user": user_email
        })

        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        print("❌ ERROR DURING PREDICTION:", str(e))
        return jsonify({'error': 'Internal server error'}), 500

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Check auth token
#         token = request.headers.get('Authorization', '').replace('Bearer ', '')
#         user_email = ''
#         if token:
#             try:
#                 decoded = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
#                 user_email = decoded.get('email', '')
#             except Exception as e:
#                 print("❌ Invalid JWT:", e)

#         # Validate file
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file uploaded'}), 400

#         file = request.files['file']
#         filename = secure_filename(file.filename)
#         allowed_extensions = {'png', 'jpg', 'jpeg'}
#         if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
#             return jsonify({'error': 'Invalid file type. Please upload .png or .jpg only'}), 400

#         # Save image
#         filepath = os.path.join('static/uploads', filename)
#         file.save(filepath)

#         # Preprocess
#         img = image.load_img(filepath, target_size=(150, 150))
#         img_tensor = image.img_to_array(img)
#         img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0

#         # Predict
#         predictions = model.predict(img_tensor)[0]
#         confidence = float(np.max(predictions)) * 100
#         predicted_class = "Pneumonia Detected" if np.argmax(predictions) == 1 else "Normal"

#         # Save to DB
#         collection.insert_one({
#             "filename": filename,
#             "prediction": predicted_class,
#             "confidence": round(confidence, 2),
#             "timestamp": datetime.now().isoformat(),
#             "user": user_email
#         })

#         return jsonify({
#             'prediction': predicted_class,
#             'confidence': round(confidence, 2)
#         })

#     except Exception as e:
#         print("❌ ERROR DURING PREDICTION:", str(e))
#         return jsonify({'error': 'Internal server error'}), 500

@app.route('/history', methods=['GET'])
def history():
    try:
        records = list(collection.find())
        for r in records:
            r['_id'] = str(r['_id'])
        return jsonify(records)
    except Exception as e:
        print("❌ ERROR FETCHING HISTORY:", str(e))
        return jsonify({'error': 'Could not fetch history'}), 500

@app.route('/feedback/<id>', methods=['POST'])
def feedback(id):
    try:
        data = request.json
        feedback_text = data.get('feedback')
        result = collection.update_one(
            {"_id": ObjectId(id)},
            {"$set": {"feedback": feedback_text}}
        )
        if result.modified_count:
            return jsonify({'message': 'Feedback saved'})
        else:
            return jsonify({'message': 'No update made'}), 404
    except Exception as e:
        print("❌ ERROR SAVING FEEDBACK:", str(e))
        return jsonify({'error': 'Could not save feedback'}), 500

@app.route('/export', methods=['GET'])
def export():
    try:
        records = collection.find()
        filename = "prediction_export.csv"
        filepath = os.path.join("static", filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID", "Filename", "Prediction", "Confidence", "Timestamp", "Feedback", "User"])
            for r in records:
                writer.writerow([
                    str(r.get('_id', '')),
                    r.get('filename', ''),
                    r.get('prediction', ''),
                    r.get('confidence', ''),
                    r.get('timestamp', ''),
                    r.get('feedback', ''),
                    r.get('user', '')
                ])

        return send_file(filepath, as_attachment=True)

    except Exception as e:
        print("❌ ERROR EXPORTING CSV:", str(e))
        return jsonify({'error': 'Could not export predictions'}), 500

# Run server
if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)



