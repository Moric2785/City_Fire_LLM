import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import os
import sys

# Allow script to be run directly: add src directory to sys.path to enable top-level package imports
try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))  
    if _SRC_DIR not in sys.path:
        sys.path.insert(0, _SRC_DIR)
except Exception:
    pass

# Import prompt generator
try:
    from .prompt_generator import FirePromptGenerator, FireIncidentData
except Exception:
    try:
        from fllm.prompt_generator import FirePromptGenerator, FireIncidentData
    except Exception:
        # Fallback placeholder for direct script execution, normal package execution won't reach here
        class FireIncidentData:  # type: ignore
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        class FirePromptGenerator:  # type: ignore
            def __init__(self):
                pass
import os
import json

class FireDataProcessor:
    """Fire incident data processing class"""
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.raw_data = None
        # Initialize prompt generator
        self.prompt_generator = FirePromptGenerator()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess data"""
        if file_path.endswith('.csv'):
            self.raw_data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            self.raw_data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
        print(f"Loaded {len(self.raw_data)} records from {file_path}")
        
        # Preprocess data and create labels
        processed_df = self.preprocess_data()
        return processed_df
        
    def preprocess_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Preprocess data"""
        if df is None:
            df = self.raw_data.copy()
            
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Normalize numerical fields
        df = self._normalize_numerical_fields(df)
        
        # Encode categorical fields
        df = self._encode_categorical_fields(df)
        
        # Create fire spread labels
        df = self.prompt_generator.create_fire_spread_labels(df)
        
        # Create distribution labels (high/medium/low)
        df = self.prompt_generator.create_distribution_labels(df)
        
        return df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        # Fill numerical fields with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                
        # Fill categorical fields with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
                
        return df
        
    def _normalize_numerical_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical fields"""
        # Normalize temperature (assuming temperature between -50 and 50)
        if 'temp' in df.columns:
            df['temp'] = df['temp'].clip(-50, 50)
            
        # Normalize humidity (0-100%)
        if 'rhum' in df.columns:
            df['rhum'] = df['rhum'].clip(0, 100)
            
        # Normalize income (assuming reasonable range)
        if 'median_income_list' in df.columns:
            df['median_income_list'] = df['median_income_list'].clip(0, 1000000)
            
        return df
        
    def _encode_categorical_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical fields"""
        # Fire origin encoding
        if 'FIRE_ORIG' in df.columns:
            fire_origin_mapping = {
                1.0: 'Kitchen',
                2.0: 'Bedroom', 
                3.0: 'Living room',
                4.0: 'Laundry area',
                5.0: 'Electrical',
                6.0: 'Heating system',
                7.0: 'Bathroom',
                8.0: 'Garage',
                9.0: 'Attic',
                10.0: 'Basement',
                11.0: 'Other'
            }
            df['fire_origin_encoded'] = df['FIRE_ORIG'].map(fire_origin_mapping).fillna('Unknown')
            
        # Building structure type encoding (based on NFIRS reference guide or codelookup)
        
        # if 'STRUC_TYPE' in df.columns:
        #     struct_type_mapping = {
        #         1: 'Enclosed Building'
        #     }
        #     df['struct_type_encoded'] = df['STRUC_TYPE'].map(struct_type_mapping).fillna('Unknown')

        # if 'STRUC_TYPE' in df.columns:
        #     struct_type_mapping = {
        #         0: 'Structure type, other',
        #         1: 'Enclosed building',
        #         2: 'Fixed portable or mobile structure',
        #         3: 'Open structure',
        #         4: 'Air supported structure',
        #         5: 'Tent',
        #         6: 'Open platform',
        #         7: 'Underground structure work areas',
        #         8: 'Connective structure'
        #     }
        #     df['struct_type_encoded'] = df['STRUC_TYPE'].map(struct_type_mapping).fillna('Unknown')
        
        # Create heat source encoding mapping (based on NFPA standards)
        if 'HEAT_SOURCE_new' in df.columns:
            heat_source_mapping = {
                '1': 'Heat from Powered Equipment',
                '2': 'Radiated, Conducted Heat from Operating Equipment',
                '3': 'Electrical Arcing',
                '4': 'Heat from Hot or Smoldering Object',
                '5': 'Explosives, Fireworks',
                '7': 'Chemical, Natural Heat Sources (e.g., Spontaneous Combustion, Lightning)',
                '8': 'Heat Spread From Another Fire',
                'UU_': 'Heat Source: Undetermined',
                'UU': 'Heat Source: Undetermined'
            }
            df['heat_source_encoded'] = df['HEAT_SOURCE_new'].map(heat_source_mapping).fillna('Unknown')
        
        # Create first ignited material encoding mapping (based on NFPA standards)
        if 'FIRST_IGN_new' in df.columns:
            first_ign_mapping = {
                '1': 'Structural Component or Finish (e.g., exterior siding, interior wall covering)',
                '2': 'Furniture and Utensils (e.g., upholstered sofa, cabinetry)',
                '3': 'Soft Goods and Wearing Apparel (e.g., mattress, bedding, clothing)',
                '4': 'Adornment, Recreational Material, Signs (e.g., Christmas tree, decoration)',
                '5': 'Storage Supplies (e.g., box, carton, pallet)',
                '6': 'General Flammable Liquids, Gases, and Chemicals',
                '7': 'Organic Materials (e.g., agricultural crops, vegetation)',
                '8': 'Other Material Compounded with Oil (e.g., linoleum, oilcloth)',
                '9': 'General Materials (e.g., books, rubbish, oily rags)',
                'UU_': 'Undetermined',
                'UU': 'Undetermined'
            }
            df['ignited_material_encoded'] = df['FIRST_IGN_new'].map(first_ign_mapping).fillna('Unknown')
        
        # Create property use encoding mapping (based on NFPA standards)
        if 'PROP_USE_new' in df.columns:
            prop_use_mapping = {
                1: 'Assembly (e.g., theater, restaurant, church)',
                2: 'Educational',
                3: 'Institutional (e.g., hospital, school, correctional facility)',
                4: 'Residential (e.g., 1- or 2-family dwelling, multifamily dwelling)',
                5: 'Mercantile, Business',
                6: 'Basic Industry, Utility, Defense',
                7: 'Manufacturing, Processing',
                8: 'Storage',
                9: 'Outside or Special Property (e.g., open land, construction site)'
            }
            df['prop_use_encoded'] = df['PROP_USE_new'].map(prop_use_mapping).fillna('Unknown')
            
        return df
        
    def convert_to_incident_data(self, df: pd.DataFrame) -> List[FireIncidentData]:
        """Convert DataFrame to FireIncidentData object list (using independent prompt generator)"""
        return self.prompt_generator.convert_dataframe_to_incident_data(df)
        
    def filter_baltimore_data(self, df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
        """Filter Baltimore single-day data"""
        # Filter Baltimore area data
        if 'STATE' in df.columns:
            # Check if data contains MD (Baltimore) or NY (New York) data
            if 'MD' in df['STATE'].values:
                # If MD data exists, prioritize MD data (Baltimore)
                filtered_data = df[df['STATE'] == 'MD'].copy()
                print(f"Filtered {len(filtered_data)} Baltimore (MD) records")
            elif 'NY' in df['STATE'].values:
                # If no MD data but NY data exists, use NY data
                filtered_data = df[df['STATE'] == 'NY'].copy()
                print(f"Filtered {len(filtered_data)} New York (NY) records")
            else:
                # If neither exists, use all data
                filtered_data = df.copy()
                print(f"No MD or NY data found, using all {len(filtered_data)} records")
        else:
            filtered_data = df.copy()
            
        # If date column is specified, filter single-day data
        if date_column and date_column in df.columns:
            # Get the most common date
            most_common_date = df[date_column].mode()[0]
            filtered_data = filtered_data[filtered_data[date_column] == most_common_date]
            
        return filtered_data
        
    def split_data(self, incident_data: List[FireIncidentData], 
                   train_ratio: float = 0.7, 
                   val_ratio: float = 0.15) -> Tuple[List[FireIncidentData], List[FireIncidentData], List[FireIncidentData]]:
        """Stratified data split into train, validation, and test sets, maintaining distribution by fire_spread_label"""
        if not incident_data:
            return [], [], []

        # Group by class
        label_to_items: Dict[int, List[FireIncidentData]] = {}
        for item in incident_data:
            label = int(getattr(item, 'fire_spread_label', 2))
            label_to_items.setdefault(label, []).append(item)

        rng = np.random.default_rng(42)
        train_data: List[FireIncidentData] = []
        val_data: List[FireIncidentData] = []
        test_data: List[FireIncidentData] = []

        for label, items in label_to_items.items():
            indices = np.arange(len(items))
            rng.shuffle(indices)
            n_total = len(indices)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            # Ensure test set exists
            if n_train + n_val >= n_total:
                n_val = max(0, n_total - n_train - 1)
            n_test = n_total - n_train - n_val

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

            train_data.extend(items[i] for i in train_idx)
            val_data.extend(items[i] for i in val_idx)
            test_data.extend(items[i] for i in test_idx)

        print(f"Data split (stratified): Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        return train_data, val_data, test_data
        
    def save_processed_data(self, incident_data: List[FireIncidentData], 
                           output_path: str = "./data/processed_incidents.json"):
        """Save processed data"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to serializable format
        serializable_data = []
        for incident in incident_data:
            incident_dict = {
                'incident_key': incident.incident_key,
                'building_info': incident.building_info,
                'weather_info': incident.weather_info,
                'fire_info': incident.fire_info,
                'community_context': incident.community_context,
                'fire_spread_label': incident.fire_spread_label
            }
            serializable_data.append(incident_dict)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
        print(f"Processed data saved to {output_path}")
        
    def load_processed_data(self, file_path: str = "./data/processed_incidents.json") -> List[FireIncidentData]:
        """Load processed data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        incident_data_list = []
        for item in data:
            incident_data = FireIncidentData(
                incident_key=item['incident_key'],
                building_info=item['building_info'],
                weather_info=item['weather_info'],
                fire_info=item['fire_info'],
                community_context=item['community_context'],
                fire_spread_label=item['fire_spread_label']
            )
            incident_data_list.append(incident_data)
            
        print(f"Loaded {len(incident_data_list)} processed incidents from {file_path}")
        return incident_data_list

    def generate_one_prompt_from_csv(self, csv_path: str) -> str:
        """Use independent prompt generator to generate one example prompt based on CSV."""
        return self.prompt_generator.generate_prompt_from_csv(csv_path)


if __name__ == "__main__":
    # Read from specified CSV and generate one example prompt using model template
    csv_path = "./data/processed_data/19_22_wo_ny_wash.csv"
    processor = FireDataProcessor()
    try:
        prompt = processor.generate_one_prompt_from_csv(csv_path)
        print(prompt)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
    except Exception as e:
        print(f"Failed to generate example prompt: {e}")
