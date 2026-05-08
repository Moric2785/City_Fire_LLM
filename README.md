# FireLLM: Fire Spread Classification with Large Language Models

FireLLM is a comprehensive machine learning system for fire spread classification using Large Language Models (LLMs) enhanced with Retrieval-Augmented Generation (RAG). The system combines the power of Llama 3.1 8B with knowledge graphs to provide accurate fire spread predictions based on fire incident data.

## 🔥 Overview

FireLLM addresses the critical need for accurate fire spread prediction in emergency response and fire safety planning. The system processes fire incident data including building characteristics, weather conditions, fire origin details, and community context to classify fire spread into three categories:

- **Class 2**: Fire within the room
- **Class 3**: Fire within the floor  
- **Class 4**: Fire beyond the floor

## ✨ Key Features

### 🧠 Advanced LLM Integration
- **Llama 3.1 8B Instruct Model**: State-of-the-art language model fine-tuned for fire classification
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning with configurable rank and alpha
- **4-bit Quantization**: Memory-efficient inference with BitsAndBytesConfig
- **Custom Classification Head**: Specialized output layer for fire spread classification

### 🔍 Retrieval-Augmented Generation (RAG)
- **Knowledge Graph Integration**: Leverages fire safety knowledge graphs for enhanced predictions
- **Semantic Retrieval**: FAISS-based vector similarity search for relevant knowledge
- **Context Augmentation**: Automatically enhances prompts with retrieved knowledge
- **Flexible Retrieval**: Configurable number of retrieved knowledge items

### 📊 Comprehensive Data Processing
- **Multi-source Data Support**: Handles various fire incident data formats (CSV, Excel)
- **Intelligent Preprocessing**: Automatic handling of missing values and data normalization
- **NFPA Standard Compliance**: Follows National Fire Protection Association coding standards
- **Stratified Data Splitting**: Maintains class distribution across train/validation/test sets

### 🎯 Advanced Training Features
- **Multiple Training Modes**: Single training, grid search, learning rate optimization
- **Hyperparameter Optimization**: Automated search for optimal model parameters
- **Early Stopping**: Prevents overfitting with configurable patience
- **Learning Rate Scheduling**: Cosine annealing with restarts for better convergence
- **Focal Loss Support**: Handles class imbalance with focal loss implementation

### 📈 Comprehensive Evaluation
- **Multi-metric Evaluation**: Accuracy, F1-score, precision, recall for each class
- **Confusion Matrix Analysis**: Detailed performance visualization
- **Error Analysis**: Comprehensive error pattern identification and reporting
- **Prediction Confidence**: Confidence scores for model predictions

## 🏗️ System Architecture

```
FireLLM/
├── src/
│   ├── fllm/                    # Core FireLLM components
│   │   ├── model_classifier.py  # Main model implementation
│   │   ├── data_processor.py    # Data preprocessing and loading
│   │   ├── prompt_generator.py  # Fire incident prompt generation
│   │   ├── trainer_classifier.py # Training pipeline
│   │   └── utils.py             # Utility functions
│   └── rag/                     # RAG system components
│       ├── knowledge_graph.py   # Knowledge graph management
│       ├── retriever.py         # Semantic retrieval system
│       ├── rag_model.py         # RAG model integration
│       ├── embedder.py          # Text embedding generation
│       └── trainer_rag.py       # RAG training pipeline
├── scripts/
│   ├── train.py                 # Main training script
│   └── train_rag.py            # RAG-specific training
├── data/
│   ├── processed_data/          # Preprocessed datasets
│   └── raw_data/               # Raw input data
├── models/                      # Trained model checkpoints
├── output/                      # Training outputs and results
└── configs/                     # Configuration files
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch transformers peft bitsandbytes
pip install faiss-cpu  # or faiss-gpu for GPU acceleration
pip install pandas numpy scikit-learn
pip install wandb  # for experiment tracking
pip install sentence-transformers  # for RAG embeddings
```

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd FLLM
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download Llama 3.1 8B model**:
```bash
# Download Llama 3.1 8B Instruct model to the specified path
# Update model path in configuration files as needed
```

### Basic Usage

#### 1. Standard Training

```bash
# Single configuration training
python scripts/train.py --mode single

# Grid search for hyperparameter optimization
python scripts/train.py --mode grid

# Learning rate search
python scripts/train.py --mode lr
```

#### 2. RAG-Enhanced Training

```bash
# Train with RAG enhancement
python scripts/train_rag.py --mode rag \
    --kg-nodes ./data/kg_nodes.csv \
    --kg-edges ./data/kg_edges.csv
```

#### 3. Model Testing

```bash
# Test trained model
python scripts/train.py --mode test \
    --model_path ./models/your_model \
    --data_path ./data/test_data.csv
```

## 📋 Data Format

### Input Data Structure

The system expects CSV files with the following key columns:

#### Fire Incident Data
- `FIRE_SPRD`: Fire spread classification (2, 3, 4, 5)
- `FIRE_ORIG`: Fire origin location code
- `AREA_ORIG`: Area of origin code
- `HEAT_SOURCE_new`: Heat source classification
- `FIRST_IGN_new`: First ignited material
- `PROP_USE_new`: Property use classification

#### Building Information
- `STRUC_TYPE`: Structure type
- `TYPE_MAT`: Construction material type
- `ACT_TAK1`: Primary action taken

#### Weather Data
- `temp`: Temperature
- `rhum`: Relative humidity
- `wind_speed`: Wind speed

#### Community Context
- `median_income_list`: Median income
- `STATE`: State code

### Example Data Processing

```python
from src.fllm.data_processor import FireDataProcessor

# Load and preprocess data
processor = FireDataProcessor()
df = processor.load_data("./data/your_fire_data.csv")

# Convert to incident data format
incident_data = processor.convert_to_incident_data(df)

# Split data for training
train_data, val_data, test_data = processor.split_data(incident_data)
```

## ⚙️ Configuration

### Model Configuration

```python
config = {
    'model_name': "/path/to/llama3-8b-instruct",
    'use_4bit': True,
    'use_lora': True,
    'lora_r': 32,
    'lora_alpha': 64,
    'lora_dropout': 0.1,
    'use_cls_head': True
}
```

### Training Configuration

```python
training_config = {
    'num_epochs': 10,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'gradient_accumulation_steps': 8,
    'weight_decay': 0.005,
    'warmup_ratio': 0.15,
    'early_stopping_patience': 8,
    'lr_scheduler': 'cosine_with_restarts'
}
```

### RAG Configuration

```python
rag_config = {
    'use_rag': True,
    'kg_nodes_path': './data/kg_nodes.csv',
    'kg_edges_path': './data/kg_edges.csv',
    'retrieval_k': 3,
    'kg_epochs': 30
}
```

## 📊 Performance Metrics

The system provides comprehensive evaluation metrics:

### Classification Metrics
- **Overall Accuracy**: Overall classification accuracy
- **Per-class Accuracy**: Accuracy for each fire spread class
- **F1-Score**: Harmonic mean of precision and recall
- **Precision/Recall**: Detailed per-class performance

### RAG Performance
- **Retrieval Success Rate**: Percentage of successful knowledge retrievals
- **Context Relevance**: Quality of retrieved knowledge
- **Prediction Enhancement**: Improvement over base model

## 🔧 Advanced Features

### Hyperparameter Optimization

```bash
# Learning rate search
python scripts/train.py --mode lr

# Grid search
python scripts/train.py --mode grid

# Focal loss optimization
python scripts/train.py --mode focal
```

### Knowledge Graph Integration

```python
from src.rag.knowledge_graph import KnowledgeGraph
from src.rag.retriever import Retriever

# Load knowledge graph
kg = KnowledgeGraph("./data/kg_nodes.csv", "./data/kg_edges.csv")

# Initialize retriever
retriever = Retriever(node_embeddings, model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### Custom Prompt Generation

```python
from src.fllm.prompt_generator import FirePromptGenerator

generator = FirePromptGenerator()
prompt = generator.create_fire_prompt(incident_data)
```

## 📁 Output Files

The system generates comprehensive output files:

### Training Outputs
- `training.log`: Detailed training logs
- `metrics.json`: Evaluation metrics
- `confusion_matrices.png`: Confusion matrix visualization
- `predictions_detailed_*.csv`: Detailed prediction results

### Model Artifacts
- `model_weights.bin`: Trained model weights
- `lora_adapter.bin`: LoRA adapter weights
- `training_summary.json`: Training configuration and results

### RAG Components
- `faiss_index.bin`: FAISS vector index
- `knowledge_graph/`: Knowledge graph files
- `retriever/`: Retrieval system components

## 🤝 Contributing

We welcome contributions to FireLLM! Please see our contributing guidelines for details on:

- Code style and standards
- Testing requirements
- Documentation updates
- Feature requests and bug reports

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Meta AI** for the Llama 3.1 model
- **Hugging Face** for the Transformers library
- **FAISS** for efficient similarity search
- **NFPA** for fire safety standards and coding systems

## 📞 Support

For questions, issues, or feature requests:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed information
3. Contact the development team

## 🔮 Future Work

- [ ] Support for additional LLM architectures
- [ ] Real-time prediction API
- [ ] Web-based user interface
- [ ] Integration with fire department systems
- [ ] Multi-language support
- [ ] Advanced visualization tools

---

**FireLLM**: Advancing fire safety through AI-powered prediction and knowledge-enhanced classification.
# City_Fire_LLM
