"""
OpenF1 API client for fetching F1 data
https://github.com/br-g/openf1
"""
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from loguru import logger
import json
import time

from app.config import OPENF1_BASE_URL, RAW_DATA_DIR


class OpenF1Client:
    """
    Client for OpenF1 API
    """
    
    def __init__(self):
        self.base_url = OPENF1_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'F1-RAG-Chatbot/1.0'
        })
    
    def _make_request(self, endpoint: str, params: Dict = None) -> List[Dict]:
        """Make API request with error handling"""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Fetched {len(data)} records from {endpoint}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from {endpoint}: {e}")
            return []
    
    def get_drivers(self, session_key: Optional[int] = None) -> List[Dict]:
        """Get drivers information"""
        params = {}
        if session_key:
            params['session_key'] = session_key
            
        return self._make_request('drivers', params)
    
    def get_sessions(self, year: Optional[int] = None, country_name: Optional[str] = None) -> List[Dict]:
        """Get sessions (races, qualifying, practice)"""
        params = {}
        if year:
            params['year'] = year
        if country_name:
            params['country_name'] = country_name
            
        return self._make_request('sessions', params)
    
    def get_race_control(self, session_key: int) -> List[Dict]:
        """Get race control messages"""
        params = {'session_key': session_key}
        return self._make_request('race_control', params)
    
    def get_lap_times(self, session_key: int, driver_number: Optional[int] = None) -> List[Dict]:
        """Get lap times"""
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
            
        return self._make_request('laps', params)
    
    def get_position_data(self, session_key: int, driver_number: Optional[int] = None) -> List[Dict]:
        """Get position data"""
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
            
        return self._make_request('position', params)
    
    def get_weather_data(self, session_key: int) -> List[Dict]:
        """Get weather data"""
        params = {'session_key': session_key}
        return self._make_request('weather', params)
    
    def get_team_radio(self, session_key: int, driver_number: Optional[int] = None) -> List[Dict]:
        """Get team radio messages"""
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
            
        return self._make_request('team_radio', params)


class F1DataProcessor:
    """
    Process F1 data for RAG knowledge base
    """
    
    def __init__(self):
        self.client = OpenF1Client()
    
    def collect_recent_race_data(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Collect recent race data and format for knowledge base
        
        Args:
            days_back: Number of days to look back for recent data
            
        Returns:
            List of formatted documents
        """
        documents = []
        
        try:
            # Get recent sessions
            current_year = datetime.now().year
            sessions = self.client.get_sessions(year=current_year)
            
            # Filter recent sessions
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            recent_sessions = []
            
            for session in sessions:
                if session.get('date_start'):
                    session_date = datetime.fromisoformat(session['date_start'].replace('Z', '+00:00'))
                    if session_date >= cutoff_date:
                        recent_sessions.append(session)
            
            logger.info(f"Found {len(recent_sessions)} recent sessions")
            
            # Process each session
            for session in recent_sessions:
                session_key = session['session_key']
                session_name = session.get('session_name', 'Unknown')
                location = session.get('location', 'Unknown')
                country = session.get('country_name', 'Unknown')
                
                # Create session overview document
                session_doc = {
                    'id': f"session_{session_key}",
                    'content': self._format_session_overview(session),
                    'metadata': {
                        'type': 'session_overview',
                        'session_key': session_key,
                        'session_name': session_name,
                        'location': location,
                        'country': country,
                        'date': session.get('date_start', ''),
                        'source': 'OpenF1 API'
                    }
                }
                documents.append(session_doc)
                
                # Get additional data for races
                if 'Race' in session_name:
                    # Race control messages
                    race_control = self.client.get_race_control(session_key)
                    if race_control:
                        rc_doc = {
                            'id': f"race_control_{session_key}",
                            'content': self._format_race_control(race_control, session),
                            'metadata': {
                                'type': 'race_control',
                                'session_key': session_key,
                                'session_name': session_name,
                                'location': location,
                                'source': 'OpenF1 API'
                            }
                        }
                        documents.append(rc_doc)
                    
                    # Weather data
                    weather = self.client.get_weather_data(session_key)
                    if weather:
                        weather_doc = {
                            'id': f"weather_{session_key}",
                            'content': self._format_weather_data(weather, session),
                            'metadata': {
                                'type': 'weather',
                                'session_key': session_key,
                                'session_name': session_name,
                                'location': location,
                                'source': 'OpenF1 API'
                            }
                        }
                        documents.append(weather_doc)
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            
            logger.info(f"Collected {len(documents)} documents from OpenF1 API")
            
        except Exception as e:
            logger.error(f"Error collecting race data: {e}")
        
        return documents
    
    def _format_session_overview(self, session: Dict) -> str:
        """Format session data as readable text"""
        content = f"""Formula 1 Session: {session.get('session_name', 'Unknown')}
Location: {session.get('location', 'Unknown')}, {session.get('country_name', 'Unknown')}
Date: {session.get('date_start', 'Unknown')}
Circuit: {session.get('circuit_short_name', 'Unknown')}
Session Type: {session.get('session_type', 'Unknown')}
Year: {session.get('year', 'Unknown')}
Meeting: {session.get('meeting_name', 'Unknown')}"""
        
        return content
    
    def _format_race_control(self, race_control: List[Dict], session: Dict) -> str:
        """Format race control messages"""
        content = f"Race Control Messages - {session.get('session_name', '')} at {session.get('location', '')}\n\n"
        
        for msg in race_control[:20]:  # Limit to first 20 messages
            time_str = msg.get('date', 'Unknown time')
            message = msg.get('message', 'No message')
            category = msg.get('category', 'General')
            
            content += f"[{time_str}] {category}: {message}\n"
        
        return content
    
    def _format_weather_data(self, weather: List[Dict], session: Dict) -> str:
        """Format weather data"""
        if not weather:
            return ""
        
        # Get representative weather data
        sample_weather = weather[0] if weather else {}
        
        content = f"""Weather Conditions - {session.get('session_name', '')} at {session.get('location', '')}

Air Temperature: {sample_weather.get('air_temperature', 'N/A')}°C
Track Temperature: {sample_weather.get('track_temperature', 'N/A')}°C
Humidity: {sample_weather.get('humidity', 'N/A')}%
Pressure: {sample_weather.get('pressure', 'N/A')} mbar
Wind Direction: {sample_weather.get('wind_direction', 'N/A')}°
Wind Speed: {sample_weather.get('wind_speed', 'N/A')} m/s
Rainfall: {'Yes' if sample_weather.get('rainfall') else 'No'}
"""
        
        return content
    
    def save_data(self, documents: List[Dict], filename: str = None):
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"openf1_data_{timestamp}.json"
            
            filepath = RAW_DATA_DIR / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(documents)} documents to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")


def main():
    """Main function for data collection"""
    processor = F1DataProcessor()
    
    # Collect recent data
    documents = processor.collect_recent_race_data(days_back=60)
    
    # Save data
    processor.save_data(documents)
    
    # Print summary
    print(f"Collected {len(documents)} documents from OpenF1 API")
    
    # Show sample
    if documents:
        print("\nSample document:")
        print(f"ID: {documents[0]['id']}")
        print(f"Type: {documents[0]['metadata']['type']}")
        print(f"Content preview: {documents[0]['content'][:200]}...")


if __name__ == "__main__":
    main()