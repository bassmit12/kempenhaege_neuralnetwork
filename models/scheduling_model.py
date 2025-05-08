import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

class SchedulingNN(nn.Module):
    """PyTorch neural network model for event scheduling"""
    
    def __init__(self, input_size):
        super(SchedulingNN, self).__init__()
        
        # Model architecture
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)

class SchedulingModel:
    def __init__(self, model_path=None):
        """
        Neural network model for recommending optimal scheduling times based on user preferences.
        
        Args:
            model_path: Path to a saved model to load, or None to create a new model
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_columns = [
            'time_preference',  # preferred hour (0-23)
            'day_preference',   # preferred day of week (0-6)
            'category_preference',  # how much user likes this category (0-1)
            # Event metrics
            'hour_of_day',     # hour (0-23)  
            'day_of_week',     # day (0-6)
            'category_id',     # One-hot encoded category
            'employee_preference',  # Employee's preference for this task (0-1)
            'task_priority',   # Priority of the task (0-1)
            'is_recurring'     # Whether the task is recurring (0-1)
        ]
        
        # Category mapping (will be updated during training)
        self.category_mapping = {}  
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self._build_model()
    
    def _build_model(self):
        """Build the neural network architecture using PyTorch"""
        self.model = SchedulingNN(len(self.feature_columns))
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def _load_model(self, model_path):
        """Load a saved PyTorch model"""
        # Load model architecture
        self.model = SchedulingNN(len(self.feature_columns))
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Also load the category mapping
        mapping_path = os.path.join(os.path.dirname(model_path), 'category_mapping.json')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                self.category_mapping = json.load(f)
    
    def save_model(self, model_path):
        """Save the model to disk"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        
        # Also save the category mapping
        mapping_path = os.path.join(os.path.dirname(model_path), 'category_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(self.category_mapping, f)
        
        print(f"Model saved to {model_path}")
        print(f"Category mapping saved to {mapping_path}")
    
    def train(self, user_preferences, event_feedback, epochs=50):
        """
        Train the model on user preferences and feedback
        
        Args:
            user_preferences: List of user preferences for event categories
            event_feedback: List of previous event feedback
            epochs: Number of training epochs
        """
        # Extract unique categories and create mapping
        categories = set()
        for pref in user_preferences:
            categories.add(pref['category_id'])
        
        # Create category mapping if needed
        if not self.category_mapping:
            self.category_mapping = {cat: idx for idx, cat in enumerate(categories)}
            
        # Prepare training data
        X, y = self._prepare_training_data(user_preferences, event_feedback)
        
        # If there's not enough data, return
        if len(X) < 2:
            print("Not enough training data")
            return None
            
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)
        
        # Create DataLoader for batching
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training history
        history = {
            'loss': [],
            'val_loss': []
        }
        
        # Set model to training mode
        self.model.train()
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Record loss
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        # Set model to evaluation mode
        self.model.eval()
        
        return history
    
    def _prepare_training_data(self, user_preferences, event_feedback):
        """
        Convert user preferences and feedback into training data
        
        Returns:
            X: Features array
            y: Target values (event ratings)
        """
        # Create a mapping from category to preferences
        preference_map = {}
        for pref in user_preferences:
            preference_map[pref['category_id']] = {
                'score': pref['preference_score'],
                'hour': pref['average_hour_preference'],
                'days': pref['preferred_days_of_week'],
                'employee_id': pref.get('employee_id', None),
                'employee_score': pref.get('employee_preference_score', 0.5),
                'is_recurring': pref.get('is_recurring', False)
            }
        
        # Create training data from feedback
        X_data = []
        y_data = []
        
        for feedback in event_feedback:
            category_id = feedback['category_id']
            if category_id not in preference_map:
                continue
                
            pref = preference_map[category_id]
            event_time = feedback['event_time']
            dt = datetime.fromisoformat(event_time)
            
            # Features
            features = [
                pref['hour'] / 23.0,  # Normalize hour to 0-1
                1.0 if dt.weekday() in pref['days'] else 0.0,  # Is preferred day
                pref['score'],  # Category preference score
                dt.hour / 23.0,  # Event hour normalized
                dt.weekday() / 6.0,  # Event day normalized
                self.category_mapping.get(category_id, 0),  # Category ID
                pref['employee_score'],  # Employee preference score
                feedback.get('priority', 0.5) / 10.0,  # Task priority normalized
                1.0 if pref['is_recurring'] else 0.0  # Is recurring task
            ]
            
            X_data.append(features)
            y_data.append(feedback['rating'])
        
        return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.float32)
    
    def predict_optimal_times(self, user_preferences, start_date, end_date, category_id):
        """
        Predict optimal times for scheduling an event
        
        Args:
            user_preferences: User's category preferences
            start_date: Start of date range (datetime)
            end_date: End of date range (datetime)
            category_id: ID of the event category
            
        Returns:
            List of (datetime, score) tuples sorted by preference score
        """
        # Find preference for this category
        category_pref = None
        for pref in user_preferences:
            if pref['category_id'] == category_id:
                category_pref = pref
                break
                
        if not category_pref:
            return []
            
        # Generate time slots to evaluate - every hour in the date range
        current = start_date
        time_slots = []
        predictions = []
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            while current <= end_date:
                # Only consider business hours (8AM-6PM)
                if 8 <= current.hour <= 18:
                    time_slots.append(current)
                    
                    # Create feature vector
                    features = [
                        category_pref['average_hour_preference'] / 23.0,
                        1.0 if current.weekday() in category_pref['preferred_days_of_week'] else 0.0,
                        category_pref['preference_score'],
                        current.hour / 23.0,
                        current.weekday() / 6.0,
                        self.category_mapping.get(category_id, 0),
                        category_pref.get('employee_preference_score', 0.5),
                        category_pref.get('priority', 0.5) / 10.0,
                        1.0 if category_pref.get('is_recurring', False) else 0.0
                    ]
                    
                    # Convert to tensor and get prediction
                    input_tensor = torch.FloatTensor([features]).to(self.device)
                    prediction = self.model(input_tensor).item()
                    predictions.append(prediction)
                
                # Advance to next hour
                current = current.replace(hour=(current.hour + 1) % 24)
                if current.hour == 0:  # New day
                    current = current.replace(day=current.day + 1)
        
        # Sort time slots by prediction score
        scored_slots = list(zip(time_slots, predictions))
        scored_slots.sort(key=lambda x: x[1], reverse=True)
        
        return scored_slots
        
    def find_best_employee(self, task, employees, existing_schedule):
        """
        Find the best employee to assign to a task
        
        Args:
            task: Task to be assigned
            employees: List of employees with preferences
            existing_schedule: Current schedule to consider
            
        Returns:
            best_employee_id, score
        """
        best_score = -1
        best_employee = None
        
        # Get the category ID from either category_id or category
        category_id = task.get('category_id', task.get('category', 'other'))
        
        for employee in employees:
            # Check if employee has the skills for this task
            if category_id not in employee.get('skills', []):
                continue
                
            # Check employee availability
            if not self._is_employee_available(employee['id'], task, existing_schedule):
                continue
                
            # Calculate score based on preference and workload
            preference_score = employee.get('category_preferences', {}).get(category_id, 0.5)
            workload_score = 1.0 - (self._get_employee_workload(employee['id'], existing_schedule) / 10.0)
            
            # Combined score
            score = (preference_score * 0.7) + (workload_score * 0.3)
            
            if score > best_score:
                best_score = score
                best_employee = employee['id']
        
        return best_employee, best_score
    
    def _is_employee_available(self, employee_id, task, existing_schedule):
        """Check if an employee is available for a task"""
        task_start = datetime.fromisoformat(task['startTime'])
        task_end = datetime.fromisoformat(task['endTime'])
        
        for event in existing_schedule:
            if event.get('employee_id') != employee_id:
                continue
                
            event_start = datetime.fromisoformat(event['startTime'])
            event_end = datetime.fromisoformat(event['endTime'])
            
            # Check for overlap
            if not (task_end <= event_start or task_start >= event_end):
                return False
                
        return True
    
    def _get_employee_workload(self, employee_id, existing_schedule):
        """Calculate current workload of an employee"""
        return sum(1 for event in existing_schedule if event.get('employee_id') == employee_id)
        
    def adjust_schedule(self, new_event, existing_schedule, user_preferences, employees):
        """
        Adjust the schedule when a new event is added
        
        Args:
            new_event: The new event being added
            existing_schedule: Current schedule
            user_preferences: User preferences
            employees: Available employees
            
        Returns:
            updated_schedule, changes
        """
        # If the new event doesn't affect existing events, just add it
        conflicts = self._find_conflicts(new_event, existing_schedule)
        if not conflicts:
            return existing_schedule + [new_event], []
            
        changes = []
        updated_schedule = [e for e in existing_schedule if e not in conflicts]
        
        # Add the new event
        updated_schedule.append(new_event)
        
        # Try to reschedule conflicting events
        for conflict in conflicts:
            # Skip recurring tasks for now as they're harder to reschedule
            if conflict.get('is_recurring', False):
                continue
                
            # Find a new time slot for this event
            category_id = conflict.get('category', 'other')
            start_date = datetime.fromisoformat(conflict['startTime']).replace(hour=8, minute=0)
            end_date = start_date.replace(hour=18)
            
            # Look for slots in the next 7 days
            for i in range(7):
                day_start = start_date.replace(day=start_date.day + i)
                day_end = end_date.replace(day=end_date.day + i)
                
                slots = self.predict_optimal_times(user_preferences, day_start, day_end, category_id)
                
                for slot_time, score in slots:
                    # Calculate event duration
                    old_start = datetime.fromisoformat(conflict['startTime'])
                    old_end = datetime.fromisoformat(conflict['endTime'])
                    duration = (old_end - old_start).seconds / 3600  # in hours
                    
                    # Create new event times
                    new_start = slot_time
                    new_end = slot_time.replace(hour=slot_time.hour + int(duration))
                    
                    # Create adjusted event
                    adjusted_event = conflict.copy()
                    adjusted_event['startTime'] = new_start.isoformat()
                    adjusted_event['endTime'] = new_end.isoformat()
                    
                    # Check if this new slot conflicts with any event in updated_schedule
                    if not self._has_conflict(adjusted_event, updated_schedule):
                        updated_schedule.append(adjusted_event)
                        changes.append({
                            'event_id': conflict['id'],
                            'old_start': conflict['startTime'],
                            'new_start': adjusted_event['startTime'],
                            'reason': 'Rescheduled due to conflict with new event'
                        })
                        break
                        
                if conflict['id'] in [c['event_id'] for c in changes]:
                    break
        
        return updated_schedule, changes
        
    def _find_conflicts(self, new_event, existing_schedule):
        """Find events that conflict with the new event"""
        conflicts = []
        
        new_start = datetime.fromisoformat(new_event['startTime'])
        new_end = datetime.fromisoformat(new_event['endTime'])
        
        for event in existing_schedule:
            event_start = datetime.fromisoformat(event['startTime'])
            event_end = datetime.fromisoformat(event['endTime'])
            
            # Check for overlap
            if not (new_end <= event_start or new_start >= event_end):
                conflicts.append(event)
                
        return conflicts
        
    def _has_conflict(self, event, schedule):
        """Check if an event conflicts with any event in the schedule"""
        if not schedule:
            return False
        
        try:
            # Safer way to check for conflicts
            conflicts = []
            
            event_start = datetime.fromisoformat(event['startTime'])
            event_end = datetime.fromisoformat(event['endTime'])
            
            for existing_event in schedule:
                try:
                    existing_start = datetime.fromisoformat(existing_event['startTime'])
                    existing_end = datetime.fromisoformat(existing_event['endTime'])
                    
                    # Check for overlap
                    if not (event_end <= existing_start or event_start >= existing_end):
                        conflicts.append(existing_event)
                except Exception as e:
                    print(f"Error checking conflict with event: {str(e)}")
                    
            return len(conflicts) > 0
        except Exception as e:
            print(f"Exception in _has_conflict: {str(e)}")
            return False
    
    def create_daily_schedule(self, recurring_tasks, employees, date):
        """
        Create a daily schedule with recurring tasks assigned to best employees
        
        Args:
            recurring_tasks: List of recurring tasks to schedule
            employees: Available employees
            date: Date to create schedule for
            
        Returns:
            daily_schedule
        """
        day_start = date.replace(hour=8, minute=0)
        day_end = date.replace(hour=18, minute=0)
        
        schedule = []
        
        # Debug information
        print(f"Creating schedule for {date.strftime('%Y-%m-%d')} with {len(recurring_tasks)} tasks")
        
        # Sort tasks by priority
        sorted_tasks = sorted(recurring_tasks, key=lambda t: t.get('priority', 5), reverse=True)
        
        for task in sorted_tasks:
            try:
                # Debug task data
                print(f"Processing task: {task.get('title')}, category: {task.get('category_id')}")
                
                # Skip if not scheduled for this day of week
                task_days = task.get('days_of_week', [])
                if date.weekday() not in task_days:
                    continue
                    
                # Find best time slot
                preferred_hour = task.get('preferred_hour', 9)
                task_duration = task.get('duration', 1)  # in hours
                
                # Try to schedule at preferred hour first
                task_start = date.replace(hour=preferred_hour, minute=0)
                task_end = date.replace(hour=preferred_hour + task_duration, minute=0)
                
                # Create event
                event = {
                    'id': f"task_{task['id']}_{date.strftime('%Y%m%d')}",
                    'title': task['title'],
                    'description': task.get('description', ''),
                    'startTime': task_start.isoformat(),
                    'endTime': task_end.isoformat(),
                    'color': task.get('color', '#4285F4'),
                    'isAllDay': False,
                    'category': task.get('category_id'),  # Store as category for compatibility
                    'is_recurring': True,
                    'task_id': task['id'],
                    'priority': task.get('priority', 5)
                }
                
                # If the preferred time has a conflict, find another slot
                if self._has_conflict(event, schedule):
                    # Try each hour of the day
                    time_slots = []
                    for hour in range(8, 19 - task_duration):
                        slot_start = date.replace(hour=hour, minute=0)
                        slot_end = date.replace(hour=hour + task_duration, minute=0)
                        
                        test_event = event.copy()
                        test_event['startTime'] = slot_start.isoformat()
                        test_event['endTime'] = slot_end.isoformat()
                        
                        if not self._has_conflict(test_event, schedule):
                            # Calculate how far this is from preferred time
                            time_diff = abs(hour - preferred_hour)
                            time_slots.append((test_event, time_diff))
                    
                    if time_slots:
                        # Choose the slot closest to preferred time
                        time_slots.sort(key=lambda x: x[1])
                        event = time_slots[0][0]
                    else:
                        # Can't schedule this task today
                        print(f"Could not find suitable time slot for task {task['title']}")
                        continue
                
                # Find best employee
                best_employee, score = self.find_best_employee(event, employees, schedule)
                if best_employee:
                    event['employee_id'] = best_employee
                    event['employee_score'] = score
                    schedule.append(event)
                    print(f"Scheduled task {task['title']} with employee {best_employee} at {event['startTime']}")
                else:
                    print(f"No suitable employee found for task {task['title']}")
            except Exception as e:
                print(f"Error processing task {task.get('title', 'unknown')}: {str(e)}")
        
        return schedule