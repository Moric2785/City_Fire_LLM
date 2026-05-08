import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

def create_sample_data(num_samples: int = 1000, save_path: str = "./data/sample_fire_data.csv") -> pd.DataFrame:
    """Create sample fire data for testing"""
    
    np.random.seed(42)
    
    # Generate sample data
    data = {
        'incident_key': [f'MD_{i:06d}_20240101_{i:04d}_0' for i in range(num_samples)],
        'state': ['MD'] * num_samples,  # Maryland (Baltimore)
        'zip_code': np.random.choice(['21201', '21202', '21205', '21206', '21207'], num_samples),
        'occupant_type': np.random.choice(['Multifamily dwellings', 'Single family dwelling', 'Commercial'], num_samples),
        'stories_above': np.random.randint(1, 6, num_samples),
        'stories_below': np.random.randint(0, 3, num_samples),
        'square_footage': np.random.randint(500, 5000, num_samples),
        'build_year': np.random.randint(1900, 2024, num_samples),
        'build_material': np.random.choice(['Brick', 'Wood', 'Vinyl', 'Concrete', 'Steel'], num_samples),
        'temperature': np.random.uniform(-10, 40, num_samples),
        'humidity': np.random.uniform(20, 90, num_samples),
        'fire_origin': np.random.choice(['Kitchen', 'Bedroom', 'Living room', 'Laundry area', 'Electrical'], num_samples),
        'fire_time': [f"{np.random.randint(0, 24):02d}h" for _ in range(num_samples)],
        'fire_date': ['1/1/2024'] * num_samples,  # Single day data
        'heat_source': np.random.choice(['Stove', 'Electrical', 'Heater', 'Other'], num_samples),
        'ignited_material': np.random.choice(['Fabric', 'Paper', 'Wood', 'Plastic', 'Food'], num_samples),
        'fire_floor': np.random.randint(1, 6, num_samples),
        'detector_status': np.random.choice(['Present', 'None', 'Unknown'], num_samples),
        'ase_status': np.random.choice(['Present', 'None', 'Unknown'], num_samples),
        'response_time': np.random.uniform(1, 15, num_samples),
        'median_income': np.random.randint(30000, 150000, num_samples),
        'median_rent': np.random.randint(800, 3000, num_samples),
        'housing_occupancy': np.random.uniform(80, 98, num_samples),
        'bachelor_degree': np.random.uniform(20, 60, num_samples),
        'elderly_population': np.random.uniform(10, 30, num_samples),
        'employment_rate': np.random.uniform(50, 90, num_samples),
        'black_population': np.random.uniform(20, 80, num_samples),
        'coal_wood_heating': np.random.uniform(0, 10, num_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create labels (simulate real situation)
    # Fire spread labels: 2, 3, 4, 5
    fire_spread_probs = [0.4, 0.3, 0.2, 0.1]  # Within room, within floor, within building, beyond building
    df['fire_spread_extent'] = np.random.choice([2, 3, 4, 5], num_samples, p=fire_spread_probs)
    
    # Casualty labels: 0, 1, 2
    casualty_probs = [0.7, 0.2, 0.1]  # No casualties, injured, fatalities
    df['casualties'] = np.random.choice([0, 1, 2], num_samples, p=casualty_probs)
    
    # Save data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"Sample data created with {num_samples} records and saved to {save_path}")
    return df

def visualize_data_distribution(df: pd.DataFrame, save_path: str = "./output/data_analysis"):
    """Visualize data distribution"""
    
    os.makedirs(save_path, exist_ok=True)
    
    # Set font for Chinese characters
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. Fire spread distribution
    plt.figure(figsize=(10, 6))
    fire_spread_counts = df['fire_spread_extent'].value_counts().sort_index()
    labels = ['Within room', 'Within floor', 'Within building', 'Beyond building']
    plt.bar(range(len(fire_spread_counts)), fire_spread_counts.values)
    plt.xlabel('Fire Spread Extent')
    plt.ylabel('Number of Incidents')
    plt.title('Fire Spread Distribution')
    plt.xticks(range(len(fire_spread_counts)), labels)
    plt.tight_layout()
    plt.savefig(f"{save_path}/fire_spread_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Casualty distribution
    plt.figure(figsize=(8, 6))
    casualty_counts = df['casualties'].value_counts().sort_index()
    casualty_labels = ['No casualties', 'Injured', 'Fatalities']
    plt.bar(range(len(casualty_counts)), casualty_counts.values)
    plt.xlabel('Casualty Status')
    plt.ylabel('Number of Incidents')
    plt.title('Casualty Distribution')
    plt.xticks(range(len(casualty_counts)), casualty_labels)
    plt.tight_layout()
    plt.savefig(f"{save_path}/casualty_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Building year distribution
    plt.figure(figsize=(12, 6))
    plt.hist(df['build_year'], bins=30, edgecolor='black')
    plt.xlabel('Building Year')
    plt.ylabel('Number of Buildings')
    plt.title('Building Year Distribution')
    plt.tight_layout()
    plt.savefig(f"{save_path}/build_year_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Building material distribution
    plt.figure(figsize=(10, 6))
    material_counts = df['build_material'].value_counts()
    plt.pie(material_counts.values, labels=material_counts.index, autopct='%1.1f%%')
    plt.title('Building Material Distribution')
    plt.tight_layout()
    plt.savefig(f"{save_path}/build_material_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Correlation heatmap
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Numerical Features Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f"{save_path}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Data visualization saved to {save_path}")

def create_config_file(config: Dict, save_path: str = "./configs/training_config.json"):
    """创建配置文件"""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        
    print(f"Configuration file saved to {save_path}")

def load_config_file(config_path: str = "./configs/training_config.json") -> Dict:
    """加载配置文件"""
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    return config

def setup_directories():
    """创建必要的目录结构"""
    
    directories = [
        "./data",
        "./models",
        "./output",
        "./configs",
        "./temp",
        "./logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def log_training_info(trainer_config: Dict, data_info: Dict, save_path: str = "./output/training_info.txt"):
    """记录训练信息"""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("FireLLM Training Information\n")
        f.write("="*50 + "\n\n")
        
        f.write("Training Configuration:\n")
        f.write("-" * 30 + "\n")
        for key, value in trainer_config.items():
            f.write(f"{key}: {value}\n")
            
        f.write(f"\nData Information:\n")
        f.write("-" * 30 + "\n")
        for key, value in data_info.items():
            f.write(f"{key}: {value}\n")
            
        f.write(f"\nTraining Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    print(f"Training information logged to {save_path}")

def calculate_model_size(model_path: str) -> Dict[str, float]:
    """计算模型大小"""
    
    total_size = 0
    file_sizes = {}
    
    for root, dirs, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            file_sizes[file] = file_size / (1024 * 1024)  # MB
            total_size += file_size
    
    total_size_mb = total_size / (1024 * 1024)
    total_size_gb = total_size_mb / 1024
    
    return {
        'total_size_mb': total_size_mb,
        'total_size_gb': total_size_gb,
        'file_sizes': file_sizes
    }

def generate_sample_prompt(incident_data: Dict) -> str:
    """生成示例提示词"""
    
    prompt = f"""<incident_key: {incident_data.get('incident_key', 'Unknown')}>
% Task Information:
You are a helpful assistant designed to forecast fire severity for a specific building.
Your task is to predict the potential probability of fire spreading to other floors from the available options:
[Fire confined in the room, Fire confined in the floor, Fire confined in the building, Fire beyond the building].

% Basic Building Information:
The building is located in {incident_data.get('state', 'Unknown')} and the zip code is {incident_data.get('zip_code', 'Unknown')}.
The building occupant type is {incident_data.get('occupant_type', 'Unknown')}, and has {incident_data.get('stories_above', 'Unknown')} stories above ground and {incident_data.get('stories_below', 'Unknown')} story below ground.
Total projected square footage is {incident_data.get('square_footage', 'Unknown')}. The building is constructed in {incident_data.get('build_year', 'Unknown')} and made by {incident_data.get('build_material', 'Unknown')}.

% Weather Information:
At the time of accident, the temperature is {incident_data.get('temperature', 'Unknown')}°C and the relative Humidity is {incident_data.get('humidity', 'Unknown')}%.

% Fire Incident Information:
The fire originated from {incident_data.get('fire_origin', 'Unknown')} at {incident_data.get('fire_time', 'Unknown')} on {incident_data.get('fire_date', 'Unknown')}.
The heat source was potentially {incident_data.get('heat_source', 'Unknown')}. The material first item ignited is {incident_data.get('ignited_material', 'Unknown')} and was located on {incident_data.get('fire_floor', 'Unknown')} floor. The detector status is: {incident_data.get('detector_status', 'Unknown')}. The ASE system status is {incident_data.get('ase_status', 'Unknown')}.
The fire department response time is estimated as {incident_data.get('response_time', 'Unknown')} min.

% Community Context:
This building is situated in a community where the median income is approximately ${incident_data.get('median_income', 'Unknown')}, 
and the median monthly rent is ${incident_data.get('median_rent', 'Unknown')}.
About {incident_data.get('housing_occupancy', 'Unknown')}% of housing units are occupied. 
Around {incident_data.get('bachelor_degree', 'Unknown')}% of the population holds a bachelor's degree or higher. 
{incident_data.get('elderly_population', 'Unknown')}% of the population is aged 62 or older, and {incident_data.get('employment_rate', 'Unknown')}% are employed. 
The Black or African American population makes up {incident_data.get('black_population', 'Unknown')}% of the community,
and {incident_data.get('coal_wood_heating', 'Unknown')}% of households use coal or wood as a heating source.

% Task Prompt:
Now, predict the potential probability of fire spreading to other floors from:
[Fire confined in the room, Fire confined in the floor, Fire confined in the building, Fire beyond the building] 
Also, predict whether the fire will produce injury and fatalities from:
[Fire without injury, Fire with injury, Fire with fatalities]

Answer: Fire spread status: <{incident_data.get('fire_spread_extent', 'Unknown')}>, Number of injuries: <{incident_data.get('casualties', 'Unknown')}>, Number of fatalities: <{incident_data.get('casualties', '0') if incident_data.get('casualties', 0) == 2 else '0'}>"""

    return prompt




