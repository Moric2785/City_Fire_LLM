# FLLM/src/rag/trainer_rag.py
import os
import json
import torch
import logging
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from tqdm import tqdm
import pandas as pd
import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Import FLLM components
from fllm.model_classifier3 import FireLLMModel
from fllm.data_processor3 import FireDataProcessor
from fllm.metrics import FireLLMMetrics
from fllm.prompt_generator3 import FireIncidentData

# Import RAG components
from .knowledge_graph import KnowledgeGraph
from .embedder import KGEmbedder
from .retriever import Retriever
from .rag_model import RAGModel


class FireLLMTrainer:
    """FireLLM model trainer (updated to support RAG)"""
    
    def __init__(self, 
                 model_name: str = "./models/llama3-8b-instruct",
                 use_wandb: bool = True,
                 project_name: str = "firellm-classifier"):
        """
        Initialize FireLLMTrainer.
        """         
        self.model_name = model_name
        self.use_wandb = use_wandb
        self.project_name = project_name
        
        # Initialize components
        self.model = None
        self.rag_model = None
        self.data_processor = FireDataProcessor()
        self.metrics_calculator = FireLLMMetrics()
        
        # Training configuration
        self.training_config = {
            'num_epochs': 15,
            'batch_size': 8,
            'learning_rate': 5e-5,
            'warmup_steps': 200,
            'weight_decay': 0.005,
            'gradient_accumulation_steps': 8,
            'save_steps': 500,
            'eval_steps': 100,
            'logging_steps': 10,
            'warmup_ratio': 0.15,
            'min_lr': 5e-7,
            'label_smoothing': 0.0,
            'early_stopping_patience': 5,
            'lr_scheduler': 'cosine_with_restarts',
            'class_weights': False,
            'dropout': 0.15,
            'focal_loss': False,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'max_grad_norm': 1.0,
            'use_cls_head': True
        }
        
        # RAG特定配置
        self.rag_config = {
            'kg_nodes_path': './data/kg_nodes.csv',
            'kg_edges_path': './data/kg_edges.csv',
            'embedding_dim': 128,
            'kg_epochs': 20,
            'kg_batch_size': 128,
            'kg_lr': 0.01,
            'retrieval_k': 5
        }
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging"""
        os.makedirs('./output', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('./output/training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_wandb(self):
        """Setup Weights & Biases"""
        if self.use_wandb and WANDB_AVAILABLE:
            try:
                wandb.init(
                    project=self.project_name,
                    config={
                        'model_name': self.model_name,
                        'training_config': self.training_config,
                        'rag_config': self.rag_config
                    }
                )
                self.logger.info("Weights & Biases initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Weights & Biases: {e}")
                self.use_wandb = False
        elif self.use_wandb and not WANDB_AVAILABLE:
            self.logger.warning("Weights & Biases not available, disabling wandb logging")
            self.use_wandb = False
            
    def load_and_prepare_data(self, 
                             data_path: str,
                             filter_baltimore: bool = True,
                             save_processed: bool = True,
                             data_cleaning: bool = True,
                             augmentation: bool = True,
                             mixup_alpha: float = 0.2) -> tuple:
        """Load and prepare data"""
        
        self.logger.debug(f"Loading data from {data_path}")
        
        processed_data = self.data_processor.load_data(data_path)
        
        if filter_baltimore:
            processed_data = self.data_processor.filter_baltimore_data(processed_data)
            
        if data_cleaning:
            self.logger.debug("Applying data cleaning...")
            processed_data = self.data_processor.preprocess_data(processed_data)
            
        incident_data = self.data_processor.convert_to_incident_data(processed_data)
        
        if augmentation:
            self.logger.debug("Data augmentation skipped - method not implemented")
        
        train_data, val_data, test_data = self.data_processor.split_data(
            incident_data, 
            train_ratio=0.7, 
            val_ratio=0.15
        )
        
        if save_processed:
            self.data_processor.save_processed_data(incident_data)
            
        self.logger.debug(f"Data prepared: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data

    def initialize_model(self, use_rag: bool = False, **kwargs):
        """
        初始化基础LLM和RAG模型（如果需要）。
        """
        self.logger.info("Initializing base FireLLMModel...")
        self.model = FireLLMModel(
            model_name=self.model_name,
            use_4bit=kwargs.get('use_4bit', True),
            use_lora=kwargs.get('use_lora', True),
            lora_r=kwargs.get('lora_r', 16),
            lora_alpha=kwargs.get('lora_alpha', 32),
            lora_dropout=kwargs.get('lora_dropout', 0.1)
        )
        self.model.load_model_and_tokenizer()
        self.logger.info("Base FireLLMModel initialized successfully.")

        if use_rag:
            self.logger.info("Initializing RAG components...")
            # 1. 加载知识图谱
            kg = KnowledgeGraph(
                nodes_path=self.rag_config['kg_nodes_path'],
                edges_path=self.rag_config['kg_edges_path']
            )
            
            # 2. 训练KG嵌入
            triplets, _, _ = kg.get_triplets()
            embedder = KGEmbedder(
                num_nodes=kg.num_nodes,
                num_relations=kg.num_relations,
                embedding_dim=self.rag_config['embedding_dim']
            )
            embedder.train(
                triplets=triplets,
                epochs=self.rag_config['kg_epochs'],
                batch_size=self.rag_config['kg_batch_size'],
                lr=self.rag_config['kg_lr']
            )
            node_embeddings = embedder.get_node_embeddings()

            # 3. 初始化Retriever
            retriever = Retriever(node_embeddings=node_embeddings)
            
            # 4. 初始化RAG模型
            self.rag_model = RAGModel(base_model=self.model, retriever=retriever, kg=kg)
            self.logger.info("RAG components initialized successfully.")

    def update_training_config(self, **kwargs):
        """更新训练配置"""
        self.training_config.update(kwargs)
        self.logger.info(f"Training config updated: {kwargs}")
        
    def update_rag_config(self, **kwargs):
        """更新RAG配置"""
        self.rag_config.update(kwargs)
        self.logger.info(f"RAG config updated: {kwargs}")

    def train_model(self, 
                   train_data: List[FireIncidentData],
                   val_data: List[FireIncidentData],
                   save_path: str = "./models/firellm_finetuned",
                   use_rag: bool = False,
                   **kwargs):
        """
        训练模型，支持标准模式和RAG模式。
        """
        if self.model is None:
            raise ValueError("Model not initialized. Please call initialize_model() first.")
        
        self.update_training_config(**kwargs)
        self.logger.info(f"Starting model training... RAG enabled: {use_rag}")
        
        train_prompts_data = train_data
        val_prompts_data = val_data

        if use_rag:
            if self.rag_model is None:
                raise ValueError("RAG model not initialized for RAG training.")
            self.logger.info("Augmenting prompts for RAG training...")
            train_prompts_data = self.rag_model.augment_incident_data_list(
                train_data, k=self.rag_config['retrieval_k']
            )
            val_prompts_data = self.rag_model.augment_incident_data_list(
                val_data, k=self.rag_config['retrieval_k']
            )
        
        # 调用基础模型的训练方法
        self.model.train(
            train_data=train_prompts_data, # 使用可能已增强的数据
            val_data=val_prompts_data,
            save_path=save_path,
            **self.training_config
        )
        self.logger.info("Model training completed.")

    def evaluate_model(self, test_data: List[FireIncidentData], use_rag: bool = False) -> Dict:
        """
        评估模型性能，支持标准模式和RAG模式。
        """
        if self.model is None:
            raise ValueError("Model not initialized.")
            
        self.logger.info(f"Evaluating on test set (size={len(test_data)}). RAG enabled: {use_rag}")
        
        # 根据是否使用RAG选择评估方式
        if use_rag:
            if self.rag_model is None:
                raise ValueError("RAG model not initialized for RAG evaluation.")
            # 使用RAG模型进行批量预测
            evaluation_results = self.model.evaluate_rag(test_data, self.rag_model)
        else:
            # 标准评估
            evaluation_results = self.model.evaluate(test_data)
        
        metrics = self.metrics_calculator.calculate_metrics(
            evaluation_results['true_labels'],
            evaluation_results['predictions']
        )
        
        self.metrics_calculator.print_metrics()
        self.metrics_calculator.save_metrics()
        
        if self.use_wandb and WANDB_AVAILABLE and getattr(wandb, 'run', None) is not None:
            wandb.log({
                'test/accuracy': metrics['overall']['accuracy'],
                'test/f1_macro': metrics['overall']['f1_macro']
            })
            
        return metrics

    def run_complete_training_pipeline(self, 
                                      data_path: str,
                                      use_rag: bool = False,
                                      save_path: str = "./models/firellm_finetuned",
                                      **kwargs):
        """
        运行完整的训练流程，支持RAG。
        """
        self.logger.info(f"Starting complete training pipeline... RAG enabled: {use_rag}")
        
        try:
            self.setup_wandb()
            
            # 更新配置
            self.update_training_config(**{k: v for k, v in kwargs.items() if k in self.training_config})
            self.update_rag_config(**{k: v for k, v in kwargs.items() if k in self.rag_config})
            
            train_data, val_data, test_data = self.load_and_prepare_data(data_path)
            
            self.initialize_model(use_rag=use_rag, **kwargs)
            
            self.train_model(train_data, val_data, save_path, use_rag=use_rag)
            
            metrics = self.evaluate_model(test_data, use_rag=use_rag)
            
            # 保存训练总结
            self._save_training_summary(metrics, save_path, use_rag)
            
            self.logger.info("Training pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
            raise
        finally:
            if self.use_wandb and WANDB_AVAILABLE and wandb.run:
                wandb.finish()

    def _save_training_summary(self, metrics: Dict, save_path: str, use_rag: bool):
        """保存训练总结"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'use_rag': use_rag,
            'training_config': self.training_config,
            'rag_config': self.rag_config if use_rag else None,
            'metrics': metrics
        }
        
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'training_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"Training summary saved to {save_path}/training_summary.json")

    def load_trained_model(self, model_path: str, use_rag: bool = False):
        """加载训练好的模型"""
        self.logger.info(f"Loading trained model from {model_path}")
        
        # 加载基础模型
        if hasattr(self.model, 'load_trained_model'):
            self.model.load_trained_model(model_path)
        else:
            self.logger.warning("Model does not support loading trained weights")
        
        # 如果使用RAG，需要重新初始化RAG组件
        if use_rag:
            self.logger.info("Reinitializing RAG components...")
            self.initialize_model(use_rag=True)
        
        self.logger.info("Model loaded successfully")

    def save_model(self, save_path: str):
        """保存模型"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存基础模型
        if hasattr(self.model, 'save_model'):
            self.model.save_model(save_path)
        
        # 保存RAG模型（如果存在）
        if self.rag_model is not None:
            self.rag_model.save_rag_model(os.path.join(save_path, 'rag_model'))
        
        self.logger.info(f"Model saved to {save_path}")

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        info = {
            'model_name': self.model_name,
            'training_config': self.training_config,
            'rag_config': self.rag_config,
            'has_rag_model': self.rag_model is not None,
            'has_base_model': self.model is not None
        }
        return info