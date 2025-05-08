from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import traceback
from datetime import datetime, timedelta
import sys
import subprocess
import requests  # Add this import for HTTP requests

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

# Additional storage files
TASKS_FILE = os.path.join(MODEL_DIR, 'recurring_tasks.json')
EMPLOYEES_FILE = os.path.join(MODEL_DIR, 'employees.json')
SCHEDULE_FILE = os.path.join(MODEL_DIR, 'schedule.json')

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

def load_tasks():
    """Load recurring tasks data"""
    if os.path.exists(TASKS_FILE):
        with open(TASKS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_tasks(tasks_data):
    """Save recurring tasks data to disk"""
    with open(TASKS_FILE, 'w') as f:
        json.dump(tasks_data, f)

def load_employees():
    """Load employees data"""
    if os.path.exists(EMPLOYEES_FILE):
        with open(EMPLOYEES_FILE, 'r') as f:
            return json.load(f)
    return []

def save_employees(employees_data):
    """Save employees data to disk"""
    with open(EMPLOYEES_FILE, 'w') as f:
        json.dump(employees_data, f)
        
def load_schedule():
    """Load current schedule data"""
    if os.path.exists(SCHEDULE_FILE):
        with open(SCHEDULE_FILE, 'r') as f:
            return json.load(f)
    return []

def save_schedule(schedule_data):
    """Save schedule data to disk"""
    with open(SCHEDULE_FILE, 'w') as f:
        json.dump(schedule_data, f)

def _get_all_preferences():
    """Helper to get all preferences as a single list"""
    all_prefs = load_preferences()
    all_user_prefs = []
    for user_pref in all_prefs:
        all_user_prefs.extend(user_pref.get('preferences', []))
    return all_user_prefs

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

@app.route('/api/employees', methods=['GET', 'POST'])
def manage_employees():
    """Manage employees and their skills/preferences"""
    try:
        if request.method == 'GET':
            employees = load_employees()
            return jsonify({
                'status': 'success',
                'employees': employees
            })
        else:  # POST
            data = request.json
            
            if 'action' not in data:
                return jsonify({'error': 'Missing action parameter'}), 400
                
            employees = load_employees()
            
            if data['action'] == 'add':
                if 'employee' not in data:
                    return jsonify({'error': 'Missing employee data'}), 400
                    
                new_employee = data['employee']
                # Check for required fields
                if 'id' not in new_employee or 'name' not in new_employee:
                    return jsonify({'error': 'Employee must have id and name'}), 400
                    
                # Check if employee already exists
                if any(emp['id'] == new_employee['id'] for emp in employees):
                    return jsonify({'error': 'Employee with this ID already exists'}), 400
                    
                employees.append(new_employee)
                save_employees(employees)
                return jsonify({
                    'status': 'success',
                    'message': 'Employee added successfully'
                })
                
            elif data['action'] == 'update':
                if 'employee' not in data:
                    return jsonify({'error': 'Missing employee data'}), 400
                    
                update_data = data['employee']
                if 'id' not in update_data:
                    return jsonify({'error': 'Employee ID is required for updates'}), 400
                    
                # Find and update employee
                for i, emp in enumerate(employees):
                    if emp['id'] == update_data['id']:
                        employees[i] = update_data
                        save_employees(employees)
                        return jsonify({
                            'status': 'success',
                            'message': 'Employee updated successfully'
                        })
                        
                return jsonify({'error': 'Employee not found'}), 404
                
            elif data['action'] == 'delete':
                if 'id' not in data:
                    return jsonify({'error': 'Employee ID is required for deletion'}), 400
                    
                # Filter out the employee to delete
                original_count = len(employees)
                employees = [emp for emp in employees if emp['id'] != data['id']]
                
                if len(employees) == original_count:
                    return jsonify({'error': 'Employee not found'}), 404
                    
                save_employees(employees)
                return jsonify({
                    'status': 'success',
                    'message': 'Employee deleted successfully'
                })
                
            else:
                return jsonify({'error': 'Invalid action'}), 400
                
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
        
@app.route('/api/tasks', methods=['GET', 'POST'])
def manage_tasks():
    """Manage recurring tasks"""
    try:
        if request.method == 'GET':
            tasks = load_tasks()
            return jsonify({
                'status': 'success',
                'tasks': tasks
            })
        else:  # POST
            data = request.json
            
            if 'action' not in data:
                return jsonify({'error': 'Missing action parameter'}), 400
                
            tasks = load_tasks()
            
            if data['action'] == 'add':
                if 'task' not in data:
                    return jsonify({'error': 'Missing task data'}), 400
                    
                new_task = data['task']
                # Check for required fields
                if not all(k in new_task for k in ['id', 'title', 'category_id', 'days_of_week']):
                    return jsonify({'error': 'Task missing required fields'}), 400
                    
                # Check if task already exists
                if any(task['id'] == new_task['id'] for task in tasks):
                    return jsonify({'error': 'Task with this ID already exists'}), 400
                    
                tasks.append(new_task)
                save_tasks(tasks)
                
                # Regenerate the schedule to include this new task
                _regenerate_schedules()
                
                return jsonify({
                    'status': 'success',
                    'message': 'Task added successfully'
                })
                
            elif data['action'] == 'update':
                if 'task' not in data:
                    return jsonify({'error': 'Missing task data'}), 400
                    
                update_data = data['task']
                if 'id' not in update_data:
                    return jsonify({'error': 'Task ID is required for updates'}), 400
                    
                # Find and update task
                for i, task in enumerate(tasks):
                    if task['id'] == update_data['id']:
                        tasks[i] = update_data
                        save_tasks(tasks)
                        
                        # Regenerate the schedule with updated task
                        _regenerate_schedules()
                        
                        return jsonify({
                            'status': 'success',
                            'message': 'Task updated successfully'
                        })
                        
                return jsonify({'error': 'Task not found'}), 404
                
            elif data['action'] == 'delete':
                if 'id' not in data:
                    return jsonify({'error': 'Task ID is required for deletion'}), 400
                    
                # Filter out the task to delete
                original_count = len(tasks)
                tasks = [task for task in tasks if task['id'] != data['id']]
                
                if len(tasks) == original_count:
                    return jsonify({'error': 'Task not found'}), 404
                    
                save_tasks(tasks)
                
                # Regenerate the schedule without this task
                _regenerate_schedules()
                
                return jsonify({
                    'status': 'success',
                    'message': 'Task deleted successfully'
                })
                
            else:
                return jsonify({'error': 'Invalid action'}), 400
                
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/schedule', methods=['GET', 'POST'])
def manage_schedule():
    """Get or update the schedule"""
    try:
        if request.method == 'GET':
            # Get date range parameters
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            
            if not start_date or not end_date:
                return jsonify({'error': 'Start and end dates are required'}), 400
                
            # Convert to datetime objects
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            
            # Load current schedule
            schedule = load_schedule()
            
            # Filter schedule for the specified date range
            filtered_schedule = []
            for event in schedule:
                event_dt = datetime.fromisoformat(event['startTime'])
                if start_dt <= event_dt <= end_dt:
                    filtered_schedule.append(event)
            
            return jsonify({
                'status': 'success',
                'schedule': filtered_schedule
            })
            
        else:  # POST - Add a new event and adjust schedule if needed
            data = request.json
            
            if 'event' not in data:
                return jsonify({'error': 'Missing event data'}), 400
                
            new_event = data['event']
            # Check for required fields
            if not all(k in new_event for k in ['id', 'title', 'startTime', 'endTime']):
                return jsonify({'error': 'Event missing required fields'}), 400
                
            # Load current schedule
            schedule = load_schedule()
            
            # Use our model to adjust the schedule
            updated_schedule, changes = model.adjust_schedule(
                new_event,
                schedule,
                _get_all_preferences(),
                load_employees()
            )
            
            # Save updated schedule
            save_schedule(updated_schedule)
            
            return jsonify({
                'status': 'success',
                'message': 'Schedule updated successfully',
                'changes': changes
            })
            
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/schedule/daily', methods=['GET'])
def get_daily_schedule():
    """Get the daily schedule with recurring tasks assigned to employees"""
    try:
        # Get date parameter
        date_str = request.args.get('date')
        
        if not date_str:
            return jsonify({'error': 'Date parameter is required'}), 400
            
        # Convert to datetime
        date = datetime.fromisoformat(date_str)
        
        # Load recurring tasks and employees
        tasks = load_tasks()
        employees = load_employees()
        
        # Generate daily schedule
        daily_schedule = model.create_daily_schedule(tasks, employees, date)
        
        return jsonify({
            'status': 'success',
            'schedule': daily_schedule
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/schedule/regenerate', methods=['POST'])
def regenerate_schedule():
    """Regenerate the schedule with recurring tasks"""
    try:
        # Regenerate schedules
        _regenerate_schedules()
        
        return jsonify({
            'status': 'success',
            'message': 'Schedule regenerated successfully'
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/schedule/create-month', methods=['POST'])
def create_month_schedule():
    """Create a schedule for a specific month with AI-assigned attendees"""
    try:
        data = request.json
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not start_date or not end_date:
            return jsonify({'error': 'Start and end dates are required'}), 400
            
        # Convert to datetime objects
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Verify this is actually for a month period
        if (end_dt - start_dt).days > 35:  # Allow slightly more than a month
            return jsonify({'error': 'Date range exceeds one month'}), 400

        # Get available employees
        employees = load_employees()
        
        # Get user preferences for scheduling
        all_prefs = _get_all_preferences()
        
        # Get event categories
        category_mapping_path = os.path.join(MODEL_DIR, 'category_mapping.json')
        if not os.path.exists(category_mapping_path):
            return jsonify({'error': 'Category mapping not found'}), 404
            
        with open(category_mapping_path, 'r') as f:
            category_mapping = json.load(f)
            
        # Generate events for each day in the date range
        events = []
        current_date = start_dt
        
        # Generate 1-3 events per day for each category
        while current_date <= end_dt:
            day_of_week = current_date.weekday()
            
            # Skip weekends unless explicitly handling them
            if day_of_week >= 5:  # 5 = Saturday, 6 = Sunday
                current_date += timedelta(days=1)
                continue
                
            for category_id, category_info in category_mapping.items():
                # Generate 1-3 events for this category on this day
                num_events = 1 + int(hash(f"{current_date.isoformat()}_{category_id}") % 3)
                
                for i in range(num_events):
                    # Choose a start time based on preferences and day
                    hour = 9 + int(hash(f"{current_date.isoformat()}_{category_id}_{i}_hour") % 8)  # Between 9am and 4pm
                    minute = [0, 15, 30, 45][int(hash(f"{current_date.isoformat()}_{category_id}_{i}_minute") % 4)]
                    
                    # Duration 30-120 minutes
                    duration_mins = 30 * (1 + int(hash(f"{current_date.isoformat()}_{category_id}_{i}_duration") % 4))
                    
                    start_time = current_date.replace(hour=hour, minute=minute)
                    end_time = start_time + timedelta(minutes=duration_mins)
                    
                    # Generate a unique ID
                    event_id = f"nn_event_{current_date.strftime('%Y%m%d')}_{category_id}_{i}"
                    
                    # Find category name from preferences
                    category_name = "Event"
                    category_color = "#3498db"  # Default blue
                    for pref in all_prefs:
                        if pref.get('category_id') == category_id:
                            category_name = pref.get('category_name', 'Event')
                            category_color = pref.get('category_color', '#3498db')
                            break
                    
                    # Create event title
                    title = f"{category_name} - {i+1}"
                    
                    # Select attendees based on the model's recommendations
                    # Choose employees with highest preference for this category/time
                    potential_attendees = []
                    for employee in employees:
                        # The employee must be available on this day
                        day_name = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'][day_of_week]
                        if not employee.get('availability', {}).get(day_name, True):
                            continue
                            
                        # Check if this employee has a preference for this category
                        has_preference = False
                        preference_score = 0.5  # Default score
                        
                        # Collect the preferences 
                        for pref in all_prefs:
                            if pref.get('category_id') == category_id:
                                has_preference = True
                                preference_score = pref.get('preference_score', 0.5)
                                break
                                
                        if has_preference:
                            # Higher score means more likely to be selected
                            potential_attendees.append({
                                'employee_id': employee.get('id'),
                                'name': employee.get('name'),
                                'score': preference_score * (1 + hash(f"{employee.get('id')}_{event_id}") % 100 / 100)
                            })
                    
                    # Sort by score and select 1-3 attendees
                    potential_attendees.sort(key=lambda x: x['score'], reverse=True)
                    num_attendees = 1 + int(hash(f"{event_id}_attendees") % min(3, len(potential_attendees)))
                    selected_attendees = potential_attendees[:num_attendees]
                    
                    # Create the event
                    event = {
                        'id': event_id,
                        'title': title,
                        'description': f"AI generated {category_name} for {current_date.strftime('%B %d, %Y')}",
                        'startTime': start_time.isoformat(),
                        'endTime': end_time.isoformat(),
                        'color': int(category_color.replace('#', '0xff'), 16) if category_color.startswith('#') else category_color,
                        'isAllDay': False,
                        'location': f"Room {1 + int(hash(event_id + '_room') % 10)}",
                        'category': category_id,
                        'attendees': [attendee['name'] for attendee in selected_attendees],
                        'scheduledBy': 'neural_network'
                    }
                    
                    events.append(event)
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Insert events into the backend database
        backend_api_url = "http://192.168.178.109:3000/api/events"
        successful_events = 0
        
        for event in events:
            try:
                response = requests.post(
                    backend_api_url,
                    json=event,
                    headers={'Content-Type': 'application/json'},
                    timeout=2
                )
                
                if response.status_code == 201:
                    successful_events += 1
                else:
                    print(f"Failed to create event: {response.text}")
            except Exception as e:
                print(f"Error sending event to backend: {e}")
        
        return jsonify({
            'status': 'success',
            'message': f'Created {successful_events} events for the next month',
            'created_events': successful_events,
            'total_events': len(events)
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
        
def hash(string):
    """Simple deterministic hash function"""
    h = 0
    for char in string:
        h = ((h * 31) + ord(char)) & 0xFFFFFFFF
    return h

@app.route('/api/seed', methods=['POST'])
def seed_data():
    """Seed the database and neural network data from backend seeders"""
    try:
        # Define backend seeders path
        backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'backend')
        seeders_dir = os.path.join(backend_dir, 'seeders')
        
        if not os.path.exists(seeders_dir):
            return jsonify({'error': f'Backend seeders directory not found at {seeders_dir}'}), 404
        
        # Run the backend seeders
        print("Running backend seeders...")
        try:
            # Run Node.js script to seed database
            seeder_script = os.path.join(seeders_dir, 'index.js')
            result = subprocess.run(['node', seeder_script], 
                                    cwd=backend_dir, 
                                    capture_output=True, 
                                    text=True, 
                                    check=True)
            print(f"Seeders output: {result.stdout}")
            
            # Extract data from backend database and save it in our format
            print("Synchronizing data with neural network files...")
            
            # Extract and convert user data
            employees_data = []
            for user in require_js_export(os.path.join(seeders_dir, 'userSeeder.js'), 'sampleUsers'):
                employees_data.append({
                    'id': user.get('id'),
                    'name': user.get('name'),
                    'email': user.get('email'),
                    'role': user.get('role'),
                    'skills': [],  # Default empty skills
                    'availability': {
                        'monday': True,
                        'tuesday': True,
                        'wednesday': True,
                        'thursday': True,
                        'friday': True,
                        'saturday': user.get('role') == 'admin',  # Admins work weekends in this example
                        'sunday': False
                    }
                })
            save_employees(employees_data)
            print(f"Saved {len(employees_data)} employees")
            
            # Extract and convert preferences data
            preferences_data = []
            users = require_js_export(os.path.join(seeders_dir, 'userSeeder.js'), 'sampleUsers')
            categories = require_js_export(os.path.join(seeders_dir, 'preferenceSeeder.js'), 'eventCategories')
            
            for user in users:
                user_prefs = []
                for category in categories:
                    # Generate similar preference data as in preferenceSeeder.js
                    pref_score = round(0.5 + (0.5 * hash(user['id'] + category['id']) % 100) / 100, 2) 
                    avg_hour = 8 + (hash(user['id'] + category['id'] + 'hour') % 10)
                    
                    # Create array of preferred days similar to preferenceSeeder.js
                    days_num = 2 + (hash(user['id'] + category['id'] + 'days') % 5)
                    all_days = list(range(7))
                    preferred_days = []
                    for _ in range(min(days_num, len(all_days))):
                        idx = hash(user['id'] + category['id'] + str(_)) % len(all_days)
                        preferred_days.append(all_days[idx])
                        all_days.pop(idx)
                    preferred_days.sort()
                    
                    user_prefs.append({
                        'category_id': category['id'],
                        'category_name': category['name'],
                        'category_color': category['color'],
                        'preference_score': pref_score,
                        'average_hour_preference': avg_hour,
                        'preferred_days_of_week': preferred_days
                    })
                
                preferences_data.append({
                    'user_id': user['id'],
                    'preferences': user_prefs,
                    'updated_at': datetime.now().isoformat()
                })
            save_preferences(preferences_data)
            print(f"Saved preferences for {len(preferences_data)} users")
            
            # Save category mapping
            category_mapping = {
                cat['id']: {'name': cat['name'], 'color': cat['color']}
                for cat in categories
            }
            save_category_mapping(category_mapping)
            print(f"Saved {len(category_mapping)} category mappings")
            
            # Generate some recurring tasks
            tasks_data = generate_recurring_tasks(categories, users)
            save_tasks(tasks_data)
            print(f"Generated {len(tasks_data)} recurring tasks")
            
            # Generate an initial schedule
            _regenerate_schedules()
            print("Generated initial schedule")
            
            return jsonify({
                'status': 'success', 
                'message': 'Database and neural network data seeded successfully',
                'stats': {
                    'employees': len(employees_data),
                    'users_with_preferences': len(preferences_data),
                    'categories': len(category_mapping),
                    'recurring_tasks': len(tasks_data)
                }
            })
            
        except subprocess.CalledProcessError as e:
            print(f"Error running seeders: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return jsonify({'error': f'Error running seeders: {e.stderr}'}), 500
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def require_js_export(file_path, export_name):
    """Extract a JavaScript module.exports value by running a small Node.js script"""
    try:
        temp_script = f"""
        const exported = require('{file_path.replace('\\\\', '\\\\\\\\')}');
        console.log(JSON.stringify(exported.{export_name}));
        """
        temp_script_path = os.path.join(os.path.dirname(file_path), '_temp_extract.js')
        
        with open(temp_script_path, 'w') as f:
            f.write(temp_script)
            
        result = subprocess.run(['node', temp_script_path], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        # Clean up temp file
        os.remove(temp_script_path)
        
        return json.loads(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error requiring JS export: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise
    except Exception as e:
        print(f"Error in require_js_export: {e}")
        raise

def hash(string):
    """Simple deterministic hash function"""
    h = 0
    for char in string:
        h = (h * 31 + ord(char)) & 0xFFFFFFFF
    return h

def save_category_mapping(category_mapping):
    """Save category mapping to disk"""
    with open(os.path.join(MODEL_DIR, 'category_mapping.json'), 'w') as f:
        json.dump(category_mapping, f)

def generate_recurring_tasks(categories, users):
    """Generate recurring tasks based on categories and users"""
    tasks = []
    task_names = {
        "meeting": ["Team Standup", "Project Review", "Client Meeting", "Strategy Session"],
        "training": ["Skills Workshop", "Learning Session", "Tutorial", "Knowledge Share"],
        "appointment": ["Checkup", "Consultation", "Review Meeting", "Assessment"],
        "personal": ["Break", "Lunch", "Personal Development", "Reflection Time"],
        "consultation": ["Customer Support", "Advisory Session", "Guidance Meeting", "Planning Session"]
    }
    
    # Generate 1-3 recurring tasks for each category
    for category in categories:
        cat_id = category['id']
        
        # Number of tasks for this category (1-3)
        num_tasks = 1 + hash(cat_id) % 3
        
        for i in range(num_tasks):
            # Get task name
            names = task_names.get(cat_id, [f"{category['name']} Task"])
            task_name = names[hash(cat_id + str(i)) % len(names)]
            
            # Generate days of week (1-5 days)
            days_count = 1 + hash(cat_id + str(i) + 'days') % 5
            all_days = list(range(7))
            days = []
            for j in range(days_count):
                idx = hash(cat_id + str(i) + str(j)) % len(all_days)
                days.append(all_days[idx])
                all_days.pop(idx)
            days.sort()
            
            # Generate start time (8am - 4pm)
            hour = 8 + hash(cat_id + str(i) + 'hour') % 9
            minute = [0, 15, 30, 45][hash(cat_id + str(i) + 'minute') % 4]
            
            # Generate duration (30-120 mins)
            duration = 30 * (1 + hash(cat_id + str(i) + 'duration') % 4)
            
            # Assign to 1-3 employees
            employees_count = 1 + hash(cat_id + str(i) + 'employees') % 3
            all_users = users.copy()
            assigned_users = []
            for j in range(min(employees_count, len(all_users))):
                idx = hash(cat_id + str(i) + str(j) + 'user') % len(all_users)
                assigned_users.append(all_users[idx]['id'])
                all_users.pop(idx)
            
            task_id = f"recurring_{cat_id}_{i}"
            
            tasks.append({
                'id': task_id,
                'title': task_name,
                'category_id': cat_id,
                'days_of_week': days,
                'start_hour': hour,
                'start_minute': minute,
                'duration_minutes': duration,
                'assigned_employees': assigned_users,
                'priority': 1 + hash(task_id + 'priority') % 3,  # Priority 1-3
                'location': f"Room {1 + hash(task_id + 'room') % 10}"
            })
    
    return tasks

def _get_all_preferences():
    """Helper to get all preferences as a single list"""
    all_prefs = load_preferences()
    all_user_prefs = []
    for user_pref in all_prefs:
        all_user_prefs.extend(user_pref.get('preferences', []))
    return all_user_prefs

def _regenerate_schedules():
    """Helper to regenerate all schedules for the next 7 days"""
    tasks = load_tasks()
    employees = load_employees()
    
    # Clear existing schedule
    schedule = []
    
    # Generate schedule for the next 7 days
    today = datetime.now()
    
    # Debug information
    print(f"Regenerating schedule with {len(tasks)} tasks and {len(employees)} employees")
    
    try:
        for i in range(7):
            date = today + timedelta(days=i)
            daily_schedule = model.create_daily_schedule(tasks, employees, date)
            schedule.extend(daily_schedule)
        
        # Save the new schedule
        save_schedule(schedule)
        
        print(f"Successfully generated schedule with {len(schedule)} events")
        return schedule
    except Exception as e:
        print(f"Error generating schedule: {str(e)}")
        print(traceback.format_exc())
        return []

if __name__ == '__main__':
    # Create a model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)