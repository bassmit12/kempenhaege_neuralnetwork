from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import traceback
from datetime import datetime
import sys

# Add the parent directory to Python path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.scheduling_model import SchedulingModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the model
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'scheduling_model')

# Load model if it exists, otherwise create a new one
model = SchedulingModel(MODEL_PATH if os.path.exists(MODEL_PATH) else None)

# Feedback storage
FEEDBACK_FILE = os.path.join(MODEL_DIR, 'feedback.json')
PREFERENCES_FILE = os.path.join(MODEL_DIR, 'preferences.json')

def load_feedback():
    """Load stored feedback data"""
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return []

def save_feedback(feedback_data):
    """Save feedback data to disk"""
    existing = load_feedback()
    existing.append(feedback_data)
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(existing, f)

def load_preferences():
    """Load stored user preferences"""
    if os.path.exists(PREFERENCES_FILE):
        with open(PREFERENCES_FILE, 'r') as f:
            return json.load(f)
    return []

def save_preferences(preferences_data):
    """Save user preferences to disk"""
    with open(PREFERENCES_FILE, 'w') as f:
        json.dump(preferences_data, f)

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Neural network API is running'
    })

@app.route('/api/preferences', methods=['POST'])
def update_preferences():
    """Store user preferences"""
    try:
        data = request.json
        user_id = data.get('user_id')
        preferences = data.get('preferences', [])
        
        if not user_id or not preferences:
            return jsonify({'error': 'Missing user_id or preferences'}), 400
            
        # Save preferences
        all_prefs = load_preferences()
        
        # Remove existing preferences for this user
        all_prefs = [p for p in all_prefs if p.get('user_id') != user_id]
        
        # Add new preferences
        user_prefs = {
            'user_id': user_id,
            'preferences': preferences,
            'updated_at': datetime.now().isoformat()
        }
        all_prefs.append(user_prefs)
        
        # Save to disk
        save_preferences(all_prefs)
        
        return jsonify({'status': 'success', 'message': 'Preferences updated'})
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit event feedback for model training"""
    try:
        data = request.json
        user_id = data.get('user_id')
        event_id = data.get('event_id')
        category_id = data.get('category_id')
        event_time = data.get('event_time')
        rating = data.get('rating')
        
        if not all([user_id, event_id, category_id, event_time, rating is not None]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Store feedback
        feedback_data = {
            'user_id': user_id,
            'event_id': event_id,
            'category_id': category_id,
            'event_time': event_time,
            'rating': float(rating),
            'submitted_at': datetime.now().isoformat()
        }
        
        save_feedback(feedback_data)
        
        # Check if we have enough feedback to retrain
        all_feedback = load_feedback()
        if len(all_feedback) % 10 == 0:  # Retrain every 10 feedback submissions
            # Get user preferences
            all_prefs = load_preferences()
            user_prefs = next((p['preferences'] for p in all_prefs if p['user_id'] == user_id), [])
            
            # Retrain model if we have preferences
            if user_prefs:
                model.train(user_prefs, all_feedback, epochs=20)
                model.save_model(MODEL_PATH)
        
        return jsonify({'status': 'success', 'message': 'Feedback submitted'})
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend_schedule():
    """Get AI recommendations for scheduling"""
    try:
        data = request.json
        user_id = data.get('user_id')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        categories = data.get('categories', [])
        
        if not all([user_id, start_date, end_date, categories]):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Convert string dates to datetime
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Get user preferences
        all_prefs = load_preferences()
        user_prefs = next((p['preferences'] for p in all_prefs if p['user_id'] == user_id), [])
        
        if not user_prefs:
            return jsonify({'error': 'No preferences found for user'}), 404
            
        # Generate recommendations for each category
        recommendations = []
        
        for category in categories:
            category_id = category.get('id')
            count = category.get('count', 1)
            
            # Skip if category not found
            if not any(p['category_id'] == category_id for p in user_prefs):
                continue
                
            # Get optimal times for this category
            optimal_times = model.predict_optimal_times(
                user_prefs,
                start_dt,
                end_dt,
                category_id
            )
            
            # Take the top 'count' recommendations
            top_recommendations = optimal_times[:count]
            
            for dt, score in top_recommendations:
                # Find category color and name
                category_info = next((p for p in user_prefs if p['category_id'] == category_id), None)
                
                recommendations.append({
                    'category_id': category_id,
                    'category_name': category_info.get('category_name', 'Unknown') if category_info else 'Unknown',
                    'category_color': category_info.get('category_color', '#808080') if category_info else '#808080',
                    'time': dt.isoformat(),
                    'score': float(score),
                    'suggested_id': f'suggested_{category_id}_{dt.strftime("%Y%m%d%H%M")}'
                })
        
        # Sort all recommendations by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Manually trigger model training"""
    try:
        # Load all feedback and preferences
        all_feedback = load_feedback()
        all_prefs = load_preferences()
        
        if not all_feedback or not all_prefs:
            return jsonify({'error': 'Not enough data for training'}), 400
            
        # Flatten all user preferences into a single list
        all_user_prefs = []
        for user_pref in all_prefs:
            all_user_prefs.extend(user_pref.get('preferences', []))
            
        # Train the model
        history = model.train(all_user_prefs, all_feedback, epochs=50)
        
        # Save the model
        model.save_model(MODEL_PATH)
        
        return jsonify({
            'status': 'success',
            'message': 'Model training complete'
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create a model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)