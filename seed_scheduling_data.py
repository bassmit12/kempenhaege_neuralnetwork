import json
import os
import sys
import uuid
from datetime import datetime, timedelta

# Add the parent directory to the path to import from api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# First, remove any existing model files to prevent compatibility issues
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'scheduling_model')
if os.path.exists(MODEL_PATH):
    print(f"Removing old model file: {MODEL_PATH}")
    os.remove(MODEL_PATH)

CATEGORY_MAPPING_PATH = os.path.join(MODEL_DIR, 'category_mapping.json')
if os.path.exists(CATEGORY_MAPPING_PATH):
    print(f"Removing old category mapping file: {CATEGORY_MAPPING_PATH}")
    os.remove(CATEGORY_MAPPING_PATH)

# Now import from server with a clean slate
from api.server import save_employees, save_tasks, save_preferences, MODEL_DIR, model

# Create sample employees
employees = [
    {
        "id": str(uuid.uuid4()),
        "name": "John Smith",
        "skills": ["meeting", "training", "personal"],
        "category_preferences": {
            "meeting": 0.9,
            "training": 0.7,
            "personal": 0.5
        },
        "availability": {
            "days_of_week": [0, 1, 2, 3, 4], # Monday to Friday
            "hours": [8, 17]  # 8 AM to 5 PM
        }
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Alice Johnson",
        "skills": ["appointment", "consultation", "personal"],
        "category_preferences": {
            "appointment": 0.8,
            "consultation": 0.9,
            "personal": 0.6
        },
        "availability": {
            "days_of_week": [0, 1, 2, 3, 4], 
            "hours": [9, 18]  # 9 AM to 6 PM
        }
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Bob Williams",
        "skills": ["meeting", "consultation", "training"],
        "category_preferences": {
            "meeting": 0.7,
            "consultation": 0.8,
            "training": 0.9
        },
        "availability": {
            "days_of_week": [0, 1, 2, 3, 4], 
            "hours": [7, 16]  # 7 AM to 4 PM
        }
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Emma Davis",
        "skills": ["appointment", "meeting", "personal"],
        "category_preferences": {
            "appointment": 0.9,
            "meeting": 0.6,
            "personal": 0.8
        },
        "availability": {
            "days_of_week": [0, 1, 2, 3, 4], 
            "hours": [10, 19]  # 10 AM to 7 PM
        }
    }
]

# Create sample recurring tasks
tasks = [
    {
        "id": str(uuid.uuid4()),
        "title": "Morning Staff Meeting",
        "description": "Daily staff coordination meeting",
        "category_id": "meeting",
        "days_of_week": [0, 1, 2, 3, 4],  # Monday to Friday
        "preferred_hour": 9,
        "duration": 1,  # 1 hour
        "color": "#4285F4",
        "priority": 8
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Patient Consultations",
        "description": "Scheduled patient consultations",
        "category_id": "consultation",
        "days_of_week": [1, 3],  # Tuesday and Thursday
        "preferred_hour": 14,
        "duration": 2,  # 2 hours
        "color": "#9C27B0",
        "priority": 9
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Staff Training",
        "description": "Weekly staff training session",
        "category_id": "training",
        "days_of_week": [2],  # Wednesday
        "preferred_hour": 13,
        "duration": 3,  # 3 hours
        "color": "#0F9D58",
        "priority": 7
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Patient Appointments",
        "description": "Regular patient check-ups",
        "category_id": "appointment",
        "days_of_week": [0, 2, 4],  # Monday, Wednesday, Friday
        "preferred_hour": 10,
        "duration": 1,  # 1 hour
        "color": "#DB4437",
        "priority": 9
    },
    {
        "id": str(uuid.uuid4()),
        "title": "Team Building",
        "description": "Team building activity",
        "category_id": "personal",
        "days_of_week": [4],  # Friday
        "preferred_hour": 15,
        "duration": 2,  # 2 hours
        "color": "#F4B400",
        "priority": 5
    }
]

# Create sample user preferences (using the same structure as before)
# But adding employee_id and is_recurring flags
user_preferences = [
    {
        "user_id": "system_scheduler",
        "preferences": [
            {
                "category_id": "meeting",
                "category_name": "Business Meetings",
                "category_color": "#4285F4",
                "preference_score": 0.8,
                "average_hour_preference": 9,
                "preferred_days_of_week": [0, 1, 2, 3, 4],
                "is_recurring": True
            },
            {
                "category_id": "training",
                "category_name": "Training Sessions",
                "category_color": "#0F9D58",
                "preference_score": 0.7,
                "average_hour_preference": 13,
                "preferred_days_of_week": [1, 2, 3],
                "is_recurring": True
            },
            {
                "category_id": "appointment",
                "category_name": "Medical Appointments",
                "category_color": "#DB4437",
                "preference_score": 0.9,
                "average_hour_preference": 10,
                "preferred_days_of_week": [0, 1, 2, 3, 4],
                "is_recurring": True
            },
            {
                "category_id": "personal",
                "category_name": "Personal Time",
                "category_color": "#F4B400",
                "preference_score": 0.6,
                "average_hour_preference": 15,
                "preferred_days_of_week": [4, 5, 6],
                "is_recurring": True
            },
            {
                "category_id": "consultation",
                "category_name": "Consultations",
                "category_color": "#9C27B0",
                "preference_score": 0.85,
                "average_hour_preference": 14,
                "preferred_days_of_week": [1, 3],
                "is_recurring": True
            }
        ],
        "updated_at": datetime.now().isoformat()
    }
]

# Create some feedback data for the model to learn from
feedback_data = []
categories = ["meeting", "training", "appointment", "personal", "consultation"]
start_date = datetime.now() - timedelta(days=30)  # Start from 30 days ago

for i in range(50):  # Generate 50 feedback entries
    category = categories[i % len(categories)]
    day_offset = i % 7  # Spread across days of week
    hour = 9 + (i % 8)  # Spread across hours 9-16
    
    # Generate datetime
    event_dt = start_date + timedelta(days=day_offset, hours=hour)
    
    # Simulate some preferences
    if category == "meeting" and hour < 12 and day_offset < 5:  # Morning meetings on weekdays are good
        rating = 0.8 + (0.2 * (10 - abs(9 - hour)) / 10)  # Better closer to 9 AM
    elif category == "training" and 12 <= hour <= 15 and 1 <= day_offset <= 3:  # Afternoon training midweek
        rating = 0.7 + (0.3 * (10 - abs(13 - hour)) / 10)  # Better closer to 1 PM
    elif category == "appointment" and 9 <= hour <= 12 and day_offset % 2 == 0:  # Morning appointments on alternating days
        rating = 0.8 + (0.2 * (10 - abs(10 - hour)) / 10)  # Better closer to 10 AM
    elif category == "personal" and hour >= 14 and day_offset >= 4:  # Late day personal time on weekends
        rating = 0.6 + (0.4 * (10 - abs(15 - hour)) / 10)  # Better closer to 3 PM
    elif category == "consultation" and 12 <= hour <= 16 and day_offset in [1, 3]:  # Midday consultations on Tue/Thu
        rating = 0.7 + (0.3 * (10 - abs(14 - hour)) / 10)  # Better closer to 2 PM
    else:
        rating = 0.3 + (0.4 * (1 - abs(13 - hour) / 10))  # Less preferred time slots
    
    # Round and ensure in range
    rating = max(0.1, min(1.0, rating))
    rating = round(rating, 2)
    
    feedback_entry = {
        "user_id": "system_scheduler",
        "event_id": f"past_event_{i}",
        "category_id": category,
        "event_time": event_dt.isoformat(),
        "rating": rating,
        "priority": 5 + (i % 5),  # Priority 5-9
        "submitted_at": datetime.now().isoformat()
    }
    
    feedback_data.append(feedback_entry)

# Save data to files
print("Saving employees data...")
save_employees(employees)

print("Saving tasks data...")
save_tasks(tasks)

print("Saving preferences data...")
with open(os.path.join(MODEL_DIR, 'preferences.json'), 'w') as f:
    json.dump(user_preferences, f)

print("Saving feedback data...")
with open(os.path.join(MODEL_DIR, 'feedback.json'), 'w') as f:
    json.dump(feedback_data, f)

# Set up category mapping
model.category_mapping = {
    "meeting": 0,
    "training": 1,
    "appointment": 2,
    "personal": 3,
    "consultation": 4
}

# Save the category mapping
with open(CATEGORY_MAPPING_PATH, 'w') as f:
    json.dump(model.category_mapping, f)

print("Sample data has been created successfully!")
print(f"\nCreated {len(employees)} employees")
print(f"Created {len(tasks)} recurring tasks")
print(f"Created {len(user_preferences)} preference profiles")
print(f"Generated {len(feedback_data)} feedback entries")
print("Setup category mapping for the model")
print("\nYou can now regenerate the schedule using the /api/schedule/regenerate endpoint")