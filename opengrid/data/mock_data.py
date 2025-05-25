"""
Mock Data Generator
Provides realistic time series data, load profiles, weather data, and operational parameters.

Author: Nik Jois (nikjois@llamasearch.ai)
License: MIT
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import json


@dataclass
class LoadProfile:
    """Load profile data structure."""
    name: str
    description: str
    data: pd.DataFrame
    peak_mw: float
    base_mw: float
    load_factor: float
    profile_type: str  # "residential", "commercial", "industrial", "mixed"


@dataclass
class WeatherData:
    """Weather data structure."""
    location: str
    data: pd.DataFrame
    climate_zone: str  # "tropical", "temperate", "arctic", "desert"
    season: str  # "spring", "summer", "fall", "winter"


@dataclass
class RenewableProfile:
    """Renewable generation profile."""
    source_type: str  # "solar", "wind", "hydro"
    capacity_mw: float
    data: pd.DataFrame
    capacity_factor: float
    variability_index: float


class MockDataGenerator:
    """Generates realistic mock data for power system analysis."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the mock data generator."""
        np.random.seed(random_seed)
        self.start_date = datetime(2024, 1, 1)
        self.time_zones = ["UTC", "EST", "PST", "CST", "MST"]
        
    def generate_load_profile(
        self,
        profile_type: str = "mixed",
        duration_days: int = 7,
        resolution_minutes: int = 15,
        peak_mw: float = 100.0,
        seasonal_variation: bool = True,
        noise_level: float = 0.05
    ) -> LoadProfile:
        """
        Generate realistic load profiles for different customer types.
        
        Args:
            profile_type: Type of load profile
            duration_days: Number of days to generate
            resolution_minutes: Time resolution in minutes
            peak_mw: Peak load in MW
            seasonal_variation: Include seasonal patterns
            noise_level: Random variation level (0-1)
        """
        periods = int(duration_days * 24 * 60 / resolution_minutes)
        times = pd.date_range(
            start=self.start_date,
            periods=periods,
            freq=f"{resolution_minutes}min"
        )
        
        # Base load patterns by type
        if profile_type == "residential":
            load_data = self._generate_residential_profile(times, peak_mw)
        elif profile_type == "commercial":
            load_data = self._generate_commercial_profile(times, peak_mw)
        elif profile_type == "industrial":
            load_data = self._generate_industrial_profile(times, peak_mw)
        else:  # mixed
            load_data = self._generate_mixed_profile(times, peak_mw)
        
        # Add seasonal variation
        if seasonal_variation:
            seasonal_factor = self._get_seasonal_factor(times)
            load_data *= seasonal_factor
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(1.0, noise_level, len(load_data))
            noise = np.clip(noise, 0.5, 1.5)  # Limit noise range
            load_data *= noise
        
        # Calculate statistics
        base_mw = np.min(load_data)
        actual_peak_mw = np.max(load_data)
        load_factor = np.mean(load_data) / actual_peak_mw
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': times,
            'load_mw': load_data,
            'hour': times.hour,
            'day_of_week': times.dayofweek,
            'month': times.month
        })
        
        return LoadProfile(
            name=f"{profile_type.title()} Load Profile",
            description=f"Synthetic {profile_type} load profile with {resolution_minutes}min resolution",
            data=df,
            peak_mw=actual_peak_mw,
            base_mw=base_mw,
            load_factor=load_factor,
            profile_type=profile_type
        )
    
    def generate_weather_data(
        self,
        location: str = "Generic",
        duration_days: int = 365,
        resolution_hours: int = 1,
        climate_zone: str = "temperate",
        season: str = "all"
    ) -> WeatherData:
        """
        Generate synthetic weather data.
        
        Args:
            location: Location name
            duration_days: Number of days
            resolution_hours: Time resolution in hours
            climate_zone: Climate zone type
            season: Season to generate
        """
        periods = int(duration_days * 24 / resolution_hours)
        times = pd.date_range(
            start=self.start_date,
            periods=periods,
            freq=f"{resolution_hours}h"
        )
        
        # Generate temperature
        temperature = self._generate_temperature(times, climate_zone, season)
        
        # Generate solar irradiance
        irradiance = self._generate_solar_irradiance(times, temperature)
        
        # Generate wind speed
        wind_speed = self._generate_wind_speed(times, temperature)
        
        # Generate humidity
        humidity = self._generate_humidity(times, temperature)
        
        # Generate precipitation
        precipitation = self._generate_precipitation(times, climate_zone)
        
        # Generate cloud cover
        cloud_cover = self._generate_cloud_cover(times, irradiance)
        
        df = pd.DataFrame({
            'timestamp': times,
            'temperature_c': temperature,
            'solar_irradiance_w_m2': irradiance,
            'wind_speed_m_s': wind_speed,
            'humidity_percent': humidity,
            'precipitation_mm': precipitation,
            'cloud_cover_percent': cloud_cover,
            'hour': times.hour,
            'day_of_year': times.dayofyear,
            'month': times.month
        })
        
        return WeatherData(
            location=location,
            data=df,
            climate_zone=climate_zone,
            season=season if season != "all" else "annual"
        )
    
    def generate_renewable_profile(
        self,
        source_type: str = "solar",
        capacity_mw: float = 50.0,
        duration_days: int = 7,
        resolution_minutes: int = 15,
        weather_data: Optional[WeatherData] = None
    ) -> RenewableProfile:
        """
        Generate renewable generation profiles.
        
        Args:
            source_type: Type of renewable source
            capacity_mw: Installed capacity in MW
            duration_days: Number of days
            resolution_minutes: Time resolution
            weather_data: Optional weather data to use
        """
        periods = int(duration_days * 24 * 60 / resolution_minutes)
        times = pd.date_range(
            start=self.start_date,
            periods=periods,
            freq=f"{resolution_minutes}min"
        )
        
        if weather_data is None:
            # Generate simplified weather for this purpose
            weather_data = self.generate_weather_data(
                duration_days=duration_days,
                resolution_hours=resolution_minutes/60
            )
        
        if source_type == "solar":
            generation = self._generate_solar_generation(times, weather_data, capacity_mw)
        elif source_type == "wind":
            generation = self._generate_wind_generation(times, weather_data, capacity_mw)
        elif source_type == "hydro":
            generation = self._generate_hydro_generation(times, capacity_mw)
        else:
            raise ValueError(f"Unknown renewable source type: {source_type}")
        
        # Calculate capacity factor and variability
        capacity_factor = np.mean(generation) / capacity_mw
        variability_index = np.std(generation) / np.mean(generation)
        
        df = pd.DataFrame({
            'timestamp': times,
            'generation_mw': generation,
            'capacity_factor': generation / capacity_mw,
            'hour': times.hour,
            'day_of_week': times.dayofweek
        })
        
        return RenewableProfile(
            source_type=source_type,
            capacity_mw=capacity_mw,
            data=df,
            capacity_factor=capacity_factor,
            variability_index=variability_index
        )
    
    def generate_price_data(
        self,
        duration_days: int = 30,
        resolution_hours: int = 1,
        base_price: float = 50.0,
        volatility: float = 0.3,
        market_type: str = "day_ahead"
    ) -> pd.DataFrame:
        """
        Generate electricity price data.
        
        Args:
            duration_days: Number of days
            resolution_hours: Time resolution
            base_price: Base price in $/MWh
            volatility: Price volatility (0-1)
            market_type: Type of market
        """
        periods = int(duration_days * 24 / resolution_hours)
        times = pd.date_range(
            start=self.start_date,
            periods=periods,
            freq=f"{resolution_hours}h"
        )
        
        # Base daily price pattern
        hour_of_day = times.hour
        daily_pattern = 1.0 + 0.3 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Weekly pattern (higher on weekdays)
        weekly_pattern = 1.0 + 0.1 * (times.dayofweek < 5)
        
        # Seasonal pattern
        seasonal_pattern = 1.0 + 0.2 * np.sin(2 * np.pi * times.dayofyear / 365)
        
        # Random component
        random_component = np.random.normal(1.0, volatility, len(times))
        random_component = np.clip(random_component, 0.1, 3.0)
        
        # Combine patterns
        prices = base_price * daily_pattern * weekly_pattern * seasonal_pattern * random_component
        
        # Occasional price spikes
        spike_probability = 0.02  # 2% chance of spike
        spikes = np.random.random(len(times)) < spike_probability
        prices[spikes] *= np.random.uniform(2, 5, np.sum(spikes))
        
        df = pd.DataFrame({
            'timestamp': times,
            'price_mwh': prices,
            'hour': times.hour,
            'day_of_week': times.dayofweek,
            'month': times.month,
            'market_type': market_type
        })
        
        return df
    
    def generate_outage_events(
        self,
        duration_days: int = 365,
        mtbf_hours: float = 8760,  # Mean time between failures
        mttr_hours: float = 4,     # Mean time to repair
        component_types: List[str] = None
    ) -> pd.DataFrame:
        """
        Generate random outage events.
        
        Args:
            duration_days: Simulation duration
            mtbf_hours: Mean time between failures
            mttr_hours: Mean time to repair
            component_types: Types of components that can fail
        """
        if component_types is None:
            component_types = ["line", "transformer", "generator", "breaker"]
        
        total_hours = duration_days * 24
        num_events = int(total_hours / mtbf_hours * len(component_types))
        
        events = []
        for i in range(num_events):
            # Random start time
            start_hour = np.random.uniform(0, total_hours)
            start_time = self.start_date + timedelta(hours=start_hour)
            
            # Random duration (exponential distribution)
            duration_hours = np.random.exponential(mttr_hours)
            end_time = start_time + timedelta(hours=duration_hours)
            
            # Random component
            component_type = np.random.choice(component_types)
            component_id = f"{component_type}_{i+1:03d}"
            
            # Severity
            severity = np.random.choice(["minor", "major", "critical"], p=[0.7, 0.25, 0.05])
            
            events.append({
                'event_id': f"outage_{i+1:05d}",
                'component_id': component_id,
                'component_type': component_type,
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': duration_hours,
                'severity': severity,
                'cause': np.random.choice([
                    "equipment_failure", "weather", "human_error",
                    "maintenance", "external_event"
                ])
            })
        
        return pd.DataFrame(events)
    
    def generate_demand_response_events(
        self,
        duration_days: int = 30,
        avg_events_per_day: float = 0.5,
        load_reduction_mw: Tuple[float, float] = (10.0, 100.0)
    ) -> pd.DataFrame:
        """
        Generate demand response events.
        
        Args:
            duration_days: Simulation duration
            avg_events_per_day: Average events per day
            load_reduction_mw: Range of load reduction (min, max)
        """
        num_events = int(duration_days * avg_events_per_day)
        
        events = []
        for i in range(num_events):
            # Random start time (typically during peak hours)
            day = np.random.randint(0, duration_days)
            hour = np.random.choice([14, 15, 16, 17, 18, 19], p=[0.1, 0.15, 0.2, 0.25, 0.2, 0.1])
            start_time = self.start_date + timedelta(days=day, hours=hour)
            
            # Duration (typically 1-4 hours)
            duration_hours = np.random.uniform(1, 4)
            end_time = start_time + timedelta(hours=duration_hours)
            
            # Load reduction
            reduction_mw = np.random.uniform(*load_reduction_mw)
            
            events.append({
                'event_id': f"dr_{i+1:04d}",
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': duration_hours,
                'load_reduction_mw': reduction_mw,
                'event_type': np.random.choice([
                    "emergency", "economic", "reliability", "test"
                ]),
                'participation_rate': np.random.uniform(0.6, 0.95)
            })
        
        return pd.DataFrame(events)
    
    def _generate_residential_profile(self, times: pd.DatetimeIndex, peak_mw: float) -> np.ndarray:
        """Generate residential load profile."""
        load = np.zeros(len(times))
        
        for i, time in enumerate(times):
            hour = time.hour
            day_of_week = time.dayofweek
            
            # Base load (always on appliances)
            base_factor = 0.3
            
            # Morning peak (6-9 AM)
            if 6 <= hour <= 9:
                morning_factor = 0.4 * (1 - abs(hour - 7.5) / 1.5)
            else:
                morning_factor = 0
            
            # Evening peak (5-9 PM)
            if 17 <= hour <= 21:
                evening_factor = 0.6 * (1 - abs(hour - 19) / 2)
            else:
                evening_factor = 0
            
            # Weekend pattern (slightly different)
            weekend_factor = 1.1 if day_of_week >= 5 else 1.0
            
            load[i] = peak_mw * (base_factor + morning_factor + evening_factor) * weekend_factor
        
        return load
    
    def _generate_commercial_profile(self, times: pd.DatetimeIndex, peak_mw: float) -> np.ndarray:
        """Generate commercial load profile."""
        load = np.zeros(len(times))
        
        for i, time in enumerate(times):
            hour = time.hour
            day_of_week = time.dayofweek
            
            # Base load (night/weekend)
            base_factor = 0.2
            
            # Business hours (8 AM - 6 PM)
            if 8 <= hour <= 18:
                business_factor = 0.8 * (1 - abs(hour - 13) / 5)
            else:
                business_factor = 0
            
            # Weekday vs weekend
            if day_of_week < 5:  # Weekday
                weekday_factor = 1.0
            else:  # Weekend
                weekday_factor = 0.3
                business_factor *= 0.5  # Reduced weekend business
            
            load[i] = peak_mw * (base_factor + business_factor * weekday_factor)
        
        return load
    
    def _generate_industrial_profile(self, times: pd.DatetimeIndex, peak_mw: float) -> np.ndarray:
        """Generate industrial load profile."""
        load = np.zeros(len(times))
        
        for i, time in enumerate(times):
            hour = time.hour
            day_of_week = time.dayofweek
            
            # Industrial plants often run continuously
            base_factor = 0.7
            
            # Production shifts
            if 6 <= hour <= 14:  # Day shift
                shift_factor = 0.3
            elif 14 <= hour <= 22:  # Evening shift
                shift_factor = 0.2
            else:  # Night shift
                shift_factor = 0.1
            
            # Reduced weekend operation
            weekend_factor = 0.8 if day_of_week >= 5 else 1.0
            
            load[i] = peak_mw * (base_factor + shift_factor) * weekend_factor
        
        return load
    
    def _generate_mixed_profile(self, times: pd.DatetimeIndex, peak_mw: float) -> np.ndarray:
        """Generate mixed load profile."""
        # Combine different customer types
        residential = self._generate_residential_profile(times, peak_mw * 0.4)
        commercial = self._generate_commercial_profile(times, peak_mw * 0.4)
        industrial = self._generate_industrial_profile(times, peak_mw * 0.2)
        
        return residential + commercial + industrial
    
    def _get_seasonal_factor(self, times: pd.DatetimeIndex) -> np.ndarray:
        """Get seasonal variation factor."""
        day_of_year = times.dayofyear
        # Summer peak (day 172 = June 21st)
        seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        return seasonal
    
    def _generate_temperature(self, times: pd.DatetimeIndex, climate: str, season: str) -> np.ndarray:
        """Generate temperature data."""
        # Base temperature by climate
        if climate == "tropical":
            base_temp = 28
            variation = 8
        elif climate == "temperate":
            base_temp = 15
            variation = 20
        elif climate == "arctic":
            base_temp = -10
            variation = 30
        else:  # desert
            base_temp = 25
            variation = 25
        
        # Seasonal variation
        day_of_year = times.dayofyear
        seasonal = variation * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Daily variation
        hour_of_day = times.hour
        daily = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Random variation
        random_var = np.random.normal(0, 2, len(times))
        
        return base_temp + seasonal + daily + random_var
    
    def _generate_solar_irradiance(self, times: pd.DatetimeIndex, temperature: np.ndarray) -> np.ndarray:
        """Generate solar irradiance data."""
        irradiance = np.zeros(len(times))
        
        for i, time in enumerate(times):
            hour = time.hour
            day_of_year = time.dayofyear
            
            # Solar elevation (simplified)
            if 6 <= hour <= 18:
                # Peak at noon
                elevation_factor = np.sin(np.pi * (hour - 6) / 12)
                # Seasonal variation
                seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                # Clear sky irradiance
                max_irradiance = 1000 * elevation_factor * seasonal_factor
                
                # Cloud effects (random)
                cloud_factor = np.random.uniform(0.3, 1.0)
                irradiance[i] = max(0, max_irradiance * cloud_factor)
            else:
                irradiance[i] = 0
        
        return irradiance
    
    def _generate_wind_speed(self, times: pd.DatetimeIndex, temperature: np.ndarray) -> np.ndarray:
        """Generate wind speed data."""
        # Base wind speed with diurnal pattern
        hour_of_day = times.hour
        base_wind = 8 + 3 * np.sin(2 * np.pi * (hour_of_day - 12) / 24)
        
        # Random variations
        random_factor = np.random.lognormal(0, 0.3, len(times))
        
        wind_speed = base_wind * random_factor
        return np.clip(wind_speed, 0, 25)  # Limit to reasonable range
    
    def _generate_humidity(self, times: pd.DatetimeIndex, temperature: np.ndarray) -> np.ndarray:
        """Generate humidity data."""
        # Inverse relationship with temperature
        base_humidity = 70 - (temperature - 20) * 1.5
        
        # Random variation
        random_var = np.random.normal(0, 10, len(times))
        
        humidity = base_humidity + random_var
        return np.clip(humidity, 10, 100)
    
    def _generate_precipitation(self, times: pd.DatetimeIndex, climate: str) -> np.ndarray:
        """Generate precipitation data."""
        if climate == "desert":
            rain_probability = 0.02
        elif climate == "tropical":
            rain_probability = 0.15
        else:
            rain_probability = 0.08
        
        # Random rain events
        rain_events = np.random.random(len(times)) < rain_probability
        precipitation = np.zeros(len(times))
        precipitation[rain_events] = np.random.exponential(5, np.sum(rain_events))
        
        return precipitation
    
    def _generate_cloud_cover(self, times: pd.DatetimeIndex, irradiance: np.ndarray) -> np.ndarray:
        """Generate cloud cover based on irradiance."""
        max_possible_irradiance = 1000 * np.maximum(0, np.sin(np.pi * (times.hour - 6) / 12))
        max_possible_irradiance[times.hour < 6] = 0
        max_possible_irradiance[times.hour > 18] = 0
        
        # Estimate cloud cover from irradiance reduction
        cloud_cover = np.zeros(len(times))
        valid_hours = max_possible_irradiance > 100
        cloud_cover[valid_hours] = 100 * (1 - irradiance[valid_hours] / max_possible_irradiance[valid_hours])
        
        return np.clip(cloud_cover, 0, 100)
    
    def _generate_solar_generation(self, times: pd.DatetimeIndex, weather: WeatherData, capacity: float) -> np.ndarray:
        """Generate solar generation based on weather."""
        # Interpolate weather data to match time resolution
        weather_interp = weather.data.set_index('timestamp').reindex(times, method='nearest')
        irradiance = weather_interp['solar_irradiance_w_m2'].values
        
        # Simple solar model: generation proportional to irradiance
        generation = capacity * irradiance / 1000  # Assume 1000 W/m² = rated capacity
        
        return np.maximum(0, generation)
    
    def _generate_wind_generation(self, times: pd.DatetimeIndex, weather: WeatherData, capacity: float) -> np.ndarray:
        """Generate wind generation based on weather."""
        # Interpolate weather data
        weather_interp = weather.data.set_index('timestamp').reindex(times, method='nearest')
        wind_speed = weather_interp['wind_speed_m_s'].values
        
        # Wind turbine power curve (simplified)
        cut_in = 3.0    # m/s
        rated = 12.0    # m/s
        cut_out = 25.0  # m/s
        
        generation = np.zeros(len(wind_speed))
        
        # Linear region (cut-in to rated)
        linear_mask = (wind_speed >= cut_in) & (wind_speed <= rated)
        generation[linear_mask] = capacity * (wind_speed[linear_mask] - cut_in) / (rated - cut_in)
        
        # Rated region (rated to cut-out)
        rated_mask = (wind_speed > rated) & (wind_speed <= cut_out)
        generation[rated_mask] = capacity
        
        return generation
    
    def _generate_hydro_generation(self, times: pd.DatetimeIndex, capacity: float) -> np.ndarray:
        """Generate hydro generation (more predictable than wind/solar)."""
        # Seasonal pattern (higher in spring due to snowmelt)
        day_of_year = times.dayofyear
        seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
        
        # Daily pattern (peak during high demand)
        hour_of_day = times.hour
        daily = 1.0 + 0.2 * np.sin(2 * np.pi * (hour_of_day - 18) / 24)
        
        # Random variation (much less than wind/solar)
        random_factor = np.random.normal(1.0, 0.1, len(times))
        
        generation = capacity * 0.6 * seasonal * daily * random_factor  # 60% capacity factor
        
        return np.maximum(0, generation) 