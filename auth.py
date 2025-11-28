from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime

auth_bp = Blueprint('auth', __name__)
SECRET_KEY = 'your-secret-key'  # store securely in production

users_db = {}  # replace with MongoDB/MySQL in real use

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.json
    if data['email'] in users_db:
        return jsonify({'error': 'Email already registered'}), 409
    users_db[data['email']] = generate_password_hash(data['password'])
    return jsonify({'message': 'User registered successfully'})

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.json
    user_pass = users_db.get(data['email'])
    if not user_pass or not check_password_hash(user_pass, data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401

    token = jwt.encode({
        'email': data['email'],
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=2)
    }, SECRET_KEY, algorithm="HS256")

    return jsonify({'token': token})
