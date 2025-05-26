from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import os
import requests
from dotenv import load_dotenv
from functools import wraps
import socket
import traceback

load_dotenv()

app = Flask(__name__)

# Configuration 
# Check if we're in Cloud Run by looking for K_SERVICE env var
IS_CLOUD_RUN = bool(os.environ.get('K_SERVICE', False))

# In Cloud Run, both frontend and backend are in the same container, so use localhost
# Otherwise, use the configured BACKEND_URL
if IS_CLOUD_RUN:
    BACKEND_URL = os.getenv('CLOUD_RUN_BACKEND_URL', 'http://127.0.0.1:8000')  
else:
    BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'No authorization token provided'}), 401
        return f(*args, **kwargs)
    return decorated

def handle_backend_error(response):
    try:
        error_data = response.json()
        return jsonify(error_data), response.status_code
    except:
        return jsonify({'error': 'Backend service unavailable'}), 503

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html')

@app.route('/search')
def search_page():
    print("[DEBUG] /search route accessed")
    return render_template('search.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/api/search', methods=['POST'])
@require_auth
def search():
    try:
        # Get search parameters from request
        data = request.form
        files = request.files.getlist('images')
        
        # Validate request
        if not data.get('query') and not files:
            return jsonify({'error': 'Either query text or images required'}), 400
            
        # Prepare the request to backend
        search_data = {
            'query': data.get('query', ''),
            'image_weight': float(data.get('image_weight', 0.7)),
            'text_weight': float(data.get('text_weight', 0.3))
        }
        
        # Prepare files if any
        files_dict = {}
        if files:
            files_dict = {
                f'images': [(file.filename, file.stream, file.content_type) 
                           for file in files if file.filename]
            }
        
        # Make request to backend
        response = requests.post(
            f'{BACKEND_URL}/search',
            data=search_data,
            files=files_dict,
            headers={'Authorization': request.headers.get('Authorization')}
        )
        
        if not response.ok:
            return handle_backend_error(response)
            
        return jsonify(response.json())
    except ValueError as e:
        return jsonify({'error': 'Invalid weight values'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['GET'])
@require_auth
def get_recommendations():
    try:
        response = requests.get(
            f'{BACKEND_URL}/recommendations',
            headers={'Authorization': request.headers.get('Authorization')}
        )
        
        if not response.ok:
            return handle_backend_error(response)
            
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
            
        data = request.get_json()
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password required'}), 400
            
        response = requests.post(
            f'{BACKEND_URL}/auth/login',
            json=data
        )
        
        if not response.ok:
            return handle_backend_error(response)
            
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
@app.route('/api/auth/register', methods=['POST'])
def auth_register():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
            
        data = request.get_json()
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password required'}), 400
            
        # Call the backend registration endpoint
        response = requests.post(
            f'{BACKEND_URL}/auth/register',
            json=data
        )
        
        if not response.ok:
            return handle_backend_error(response)
            
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
@require_auth
def auth_logout():
    try:
        response = requests.post(
            f'{BACKEND_URL}/auth/logout',
            headers={'Authorization': request.headers.get('Authorization')}
        )
        return jsonify({'message': 'Logged out successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cart', methods=['GET'])
@require_auth
def get_cart():
    try:
        response = requests.get(
            f'{BACKEND_URL}/cart',
            headers={'Authorization': request.headers.get('Authorization')}
        )
        
        if not response.ok:
            return handle_backend_error(response)
            
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cart/add', methods=['POST'])
@require_auth
def add_to_cart():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
            
        data = request.get_json()
        if not data.get('product_id') or not isinstance(data.get('quantity', 0), int):
            return jsonify({'error': 'Valid product_id and quantity required'}), 400
            
        response = requests.post(
            f'{BACKEND_URL}/cart/items',
            json=data,
            headers={'Authorization': request.headers.get('Authorization')}
        )
        
        if not response.ok:
            return handle_backend_error(response)
            
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/checkout', methods=['POST'])
@require_auth
def checkout():
    try:
        response = requests.post(
            f'{BACKEND_URL}/cart/checkout',
            headers={'Authorization': request.headers.get('Authorization')}
        )
        
        if not response.ok:
            return handle_backend_error(response)
            
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    tb = traceback.format_exc()
    print(f"[ERROR] Unhandled Exception: {tb}")
    return f"<pre>{tb}</pre>", 500

@app.route('/healthz')
def healthz():
    return "OK", 200

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))