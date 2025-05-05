import json
import numpy as np
from datetime import datetime, timedelta

def parse_iso_datetime(datetime_str):
    """Parse ISO format datetime string into datetime object"""
    return datetime.fromisoformat(datetime_str)

def normalize_hour(hour):
    """Normalize hour value to range 0-1"""
    return hour / 23.0

def normalize_day(day):
    """Normalize day of week to range 0-1"""
    return day / 6.0

def calculate_preferred_hour(time_ratings):
    """Calculate the preferred hour based on ratings throughout the day
    
    Args:
        time_ratings: List of (hour, rating) tuples
    
    Returns:
        Preferred hour (0-23)
    """
    if not time_ratings:
        return 9  # Default to 9AM if no data
    
    # Weight each hour by its rating
    weighted_sum = sum(hour * rating for hour, rating in time_ratings)
    total_weight = sum(rating for _, rating in time_ratings)
    
    if total_weight == 0:
        return 9
    
    # Calculate weighted average
    preferred_hour = round(weighted_sum / total_weight)
    
    # Ensure value is in valid range
    return max(0, min(23, preferred_hour))

def calculate_preferred_days(day_ratings):
    """Calculate preferred days based on ratings
    
    Args:
        day_ratings: List of (day, rating) tuples
            where day is 0-6 (Monday-Sunday)
    
    Returns:
        List of preferred days (indices 0-6)
    """
    # Default to weekdays if no data
    if not day_ratings:
        return [0, 1, 2, 3, 4]
    
    # Select days with rating > 0.5
    preferred = [day for day, rating in day_ratings if rating > 0.5]
    
    # If nothing selected, take the top 3 highest rated days
    if not preferred:
        sorted_days = sorted(day_ratings, key=lambda x: x[1], reverse=True)
        preferred = [day for day, _ in sorted_days[:3]]
    
    return preferred

def generate_event_title(category_name, time):
    """Generate a title for suggested events
    
    Args:
        category_name: Name of the event category
        time: Datetime for the event
    
    Returns:
        Generated title string
    """
    weekday = time.strftime("%A")
    time_str = time.strftime("%I:%M %p")
    
    return f"{category_name} - {weekday} at {time_str}"

def format_api_response(data):
    """Format API response to be more readable
    
    Args:
        data: API response data
        
    Returns:
        Formatted string
    """
    return json.dumps(data, indent=2)