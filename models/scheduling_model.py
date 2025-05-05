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
                'days': pref['preferred_days_of_week']
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
                self.category_mapping.get(category_id, 0)  # Category ID
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
                        self.category_mapping.get(category_id, 0)
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
    
    def save_model(self, model_path):
        """Save the model to disk"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        
        # Also save the category mapping
        mapping_path = os.path.join(os.path.dirname(model_path), 'category_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(self.category_mapping, f)