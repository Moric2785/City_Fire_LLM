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
from .model_classifier import FireLLMModel
from .data_processor import FireDataProcessor
from .metrics import FireLLMMetrics
from .prompt_generator import FireIncidentData

class FireLLMTrainer:
    """FireLLM model trainer"""
    
    def __init__(self, 
                 model_name: str = "./models/llama3.1-8b-instruct",
                 use_wandb: bool = True,
                 project_name: str = "firellm-classifier"):
        
        self.model_name = model_name
        self.use_wandb = use_wandb
        self.project_name = project_name
        
        # Initialize components
        self.model = None
        self.data_processor = FireDataProcessor()
        self.metrics_calculator = FireLLMMetrics()
        
        # Training configuration - optimized for 3-class classification
        self.training_config = {
            'num_epochs': 15,
            'batch_size': 8,  # Reduce batch size to increase gradient update frequency
            'learning_rate': 5e-5,  # Lower learning rate for stability
            'warmup_steps': 200,
            'weight_decay': 0.005,  # Reduce regularization
            'gradient_accumulation_steps': 8,  # Increase gradient accumulation to maintain effective batch size
            'save_steps': 500,
            'eval_steps': 100,
            'logging_steps': 10,
            'warmup_ratio': 0.15,
            'min_lr': 5e-7,
            'label_smoothing': 0.0,
            'early_stopping_patience': 5,  # Increase patience
            'lr_scheduler': 'cosine_with_restarts',  # Use cosine scheduler with restarts
            'class_weights': False,  # 3-class classification, relatively balanced data
            'dropout': 0.15,  # Moderate dropout
            'focal_loss': False,  # 3-class classification, disable focal loss
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'max_grad_norm': 1.0,  # Gradient clipping
            'use_cls_head': True
        }
        
        # Setup logging
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
                        'training_config': self.training_config
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
        
        # Load and preprocess data (load_data already includes preprocessing)
        processed_data = self.data_processor.load_data(data_path)
        
        # Filter Baltimore data (if needed)
        if filter_baltimore:
            processed_data = self.data_processor.filter_baltimore_data(processed_data)
            
        # Data cleaning (if needed)
        if data_cleaning:
            self.logger.debug("Applying data cleaning...")
            processed_data = self.data_processor.preprocess_data(processed_data)
            
        # Convert to FireIncidentData objects
        incident_data = self.data_processor.convert_to_incident_data(processed_data)
        
        # Data augmentation (if needed)
        if augmentation:
            self.logger.debug("Applying data augmentation...")
            # Skip data augmentation for now as method doesn't exist
            # incident_data = self.data_processor.augment_data(incident_data, mixup_alpha=mixup_alpha)
            self.logger.debug("Data augmentation skipped - method not implemented")
        
        # Split data
        train_data, val_data, test_data = self.data_processor.split_data(
            incident_data, 
            train_ratio=0.7, 
            val_ratio=0.15
        )
        
        # Save processed data
        if save_processed:
            self.data_processor.save_processed_data(incident_data)
            
        self.logger.debug(f"Data prepared: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
        
    def initialize_model(self, 
                        use_4bit: bool = True,
                        use_lora: bool = True,
                        lora_r: int = 16,
                        lora_alpha: int = 32,
                        lora_dropout: float = 0.1,
                        use_gradient_checkpointing: bool = True,
                        dropout: float = 0.2,
                        weight_decay: float = 0.05):
        """初始化模型"""
        
        self.logger.debug(f"Initializing model: {self.model_name}")
        
        self.model = FireLLMModel(
            model_name=self.model_name,
            use_4bit=use_4bit,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        # 将训练配置同步到模型，作为默认超参数来源
        try:
            setattr(self.model, 'training_config', {
                'num_epochs': self.training_config.get('num_epochs', 3),
                'batch_size': self.training_config.get('batch_size', 4),
                'learning_rate': self.training_config.get('learning_rate', 2e-4),
                'save_path': './models/firellm_finetuned'
            })
        except Exception:
            pass

        # 加载模型和分词器
        self.model.load_model_and_tokenizer()
        
        # 确保模型在正确的设备上
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'peft_config'):
            # 如果是LoRA模型，确保所有参数都在正确的设备上
            self.model.model = self.model.model.to(self.model.device)
        
        self.logger.debug("Model initialized successfully")
        
    def update_training_config(self, **kwargs):
        """更新训练配置"""
        self.training_config.update(kwargs)
        self.logger.info(f"Training config updated: {kwargs}")
        
    def train_model(self, 
                   train_data: List[FireIncidentData],
                   val_data: List[FireIncidentData],
                   save_path: str = "./models/firellm_finetuned",
                   num_epochs: int = None,
                   batch_size: int = None,
                   learning_rate: float = None,
                   lr_scheduler: str = 'cosine',
                   warmup_ratio: float = 0.1,
                   min_lr: float = 1e-6,
                   label_smoothing: float = 0.05,
                   gradient_accumulation_steps: int = 4,
                   early_stopping_patience: int = 3,
                   class_weights: bool = True,
                   weight_decay: float = 0.001,
                   dropout: float = 0.2,
                   **kwargs):
        """训练模型"""
        
        if self.model is None:
            raise ValueError("Model not initialized. Please call initialize_model() first.")
            
        # 更新训练配置
        training_params = {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'lr_scheduler': lr_scheduler,
            'warmup_ratio': warmup_ratio,
            'min_lr': min_lr,
            'label_smoothing': label_smoothing,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'early_stopping_patience': early_stopping_patience,
            'class_weights': class_weights,
            'weight_decay': weight_decay,
            'dropout': dropout
        }
        
        # 过滤掉None值
        training_params = {k: v for k, v in training_params.items() if v is not None}
        
        if training_params:
            self.update_training_config(**training_params)
            
        if kwargs:
            self.update_training_config(**kwargs)
            
        self.logger.debug("Starting model training...")
        self.logger.debug(f"Training config: {self.training_config}")
        
        # 记录训练开始时间
        start_time = datetime.now()
        
        try:
            # 开始训练
            # 将最新training_config同步至模型
            try:
                if hasattr(self, 'training_config'):
                    current_cfg = getattr(self, 'training_config')
                    if not hasattr(self.model, 'training_config'):
                        self.model.training_config = {}
                    self.model.training_config.update({
                        'num_epochs': current_cfg.get('num_epochs'),
                        'batch_size': current_cfg.get('batch_size'),
                        'learning_rate': current_cfg.get('learning_rate'),
                        'lr_scheduler': current_cfg.get('lr_scheduler', 'cosine'),
                        'warmup_ratio': current_cfg.get('warmup_ratio', 0.1),
                        'min_lr': current_cfg.get('min_lr', 1e-6),
                        'label_smoothing': current_cfg.get('label_smoothing', 0.05),
                        'gradient_accumulation_steps': current_cfg.get('gradient_accumulation_steps', 4),
                        'early_stopping_patience': current_cfg.get('early_stopping_patience', 3),
                        'class_weights': current_cfg.get('class_weights', True),
                        'weight_decay': current_cfg.get('weight_decay', 0.001),
                        'dropout': current_cfg.get('dropout', 0.2),
                        'save_path': save_path
                    })
            except Exception:
                pass

            # 允许显式传参覆盖，否则由模型内部从training_config取默认
            self.model.train(
                train_data=train_data,
                val_data=val_data,
                num_epochs=kwargs.get('num_epochs'),
                batch_size=kwargs.get('batch_size'),
                learning_rate=kwargs.get('learning_rate'),
                save_path=save_path,
                # 传递所有新的优化参数
                weight_decay=kwargs.get('weight_decay'),
                gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps'),
                lr_scheduler=kwargs.get('lr_scheduler'),
                warmup_ratio=kwargs.get('warmup_ratio'),
                min_lr=kwargs.get('min_lr'),
                early_stopping_patience=kwargs.get('early_stopping_patience'),
                dropout=kwargs.get('dropout'),
                # 焦点损失参数
                focal_loss=kwargs.get('focal_loss'),
                focal_alpha=kwargs.get('focal_alpha'),
                focal_gamma=kwargs.get('focal_gamma'),
                max_grad_norm=kwargs.get('max_grad_norm'),
                use_cls_head=kwargs.get('use_cls_head')
            )
            
            # 记录训练结束时间
            end_time = datetime.now()
            training_duration = end_time - start_time
            
            self.logger.debug(f"Training completed successfully in {training_duration}")
            
            # 记录到wandb（仅当已初始化）
            if self.use_wandb and WANDB_AVAILABLE and getattr(wandb, 'run', None) is not None:
                try:
                    wandb.log({
                        'training_duration_minutes': training_duration.total_seconds() / 60,
                        'final_model_path': save_path,
                        'total_epochs': self.training_config['num_epochs'],
                        'final_train_loss': 'completed',
                        'model_save_path': save_path
                    })
                except Exception as e:
                    self.logger.warning(f"wandb.log skipped: {e}")
                
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
            
    def evaluate_model(self, test_data: List[FireIncidentData]) -> Dict:
        """评估模型性能"""
        
        if self.model is None:
            raise ValueError("Model not initialized. Please call initialize_model() first.")
            
        self.logger.info(f"Evaluating on test set: size={len(test_data)}")
        
        # 进行预测
        self.logger.debug("Running predictions on test set...")
        evaluation_results = self.model.evaluate(test_data)
        
        # 计算详细指标
        self.logger.debug("Computing evaluation metrics...")
        metrics = self.metrics_calculator.calculate_metrics(
            evaluation_results['true_labels'],
            evaluation_results['predictions']
        )
        
        # 打印指标
        self.logger.debug("Generating evaluation report...")
        self.metrics_calculator.print_metrics()
        
        # 保存指标
        self.logger.debug("Saving evaluation artifacts...")
        self.metrics_calculator.save_metrics()
        self.metrics_calculator.generate_detailed_report()
        
        # 保存详细预测结果到CSV
        self.logger.debug("Saving detailed prediction CSV...")
        self._save_predictions_to_csv(evaluation_results, test_data)
        
        # 绘制混淆矩阵
        self.logger.debug("Generating confusion matrices...")
        self.metrics_calculator.plot_confusion_matrices()
        
        # 记录到wandb
        if self.use_wandb and WANDB_AVAILABLE and getattr(wandb, 'run', None) is not None:
            try:
                wandb.log({
                    'fire_spread_accuracy': metrics['fire_spread']['accuracy'],
                    'overall_accuracy': metrics['overall']['accuracy']
                })
            except Exception as e:
                self.logger.warning(f"wandb.log failed: {e}")
            
        return metrics
        
    def _save_predictions_to_csv(self, evaluation_results: Dict, test_data: List[FireIncidentData]):
        """保存详细预测结果到CSV文件"""
        import pandas as pd
        from datetime import datetime
        
        # 准备数据
        rows = []
        for i, (pred, true_label) in enumerate(zip(evaluation_results['predictions'], evaluation_results['true_labels'])):
            row = {
                'sample_id': i + 1,
                'incident_key': test_data[i].incident_key,
                'fire_spread_true': true_label['fire_spread'],
                'fire_spread_pred': pred['fire_spread']['predicted_class'],
                'fire_spread_pred_name': pred['fire_spread']['class_name'],
                'fire_spread_correct': true_label['fire_spread'] == pred['fire_spread']['predicted_class'],

                'fire_spread_prob_2': evaluation_results['predictions'][i]['fire_spread']['probabilities'].get(2, 0.0),
                'fire_spread_prob_3': evaluation_results['predictions'][i]['fire_spread']['probabilities'].get(3, 0.0),
                'fire_spread_prob_4': evaluation_results['predictions'][i]['fire_spread']['probabilities'].get(4, 0.0),
                'overall_correct': true_label['fire_spread'] == pred['fire_spread']['predicted_class']
            }
            rows.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(rows)
        
        # 保存到CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"./output/predictions_detailed_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # 打印统计信息
        total_samples = len(df)
        fire_spread_accuracy = df['fire_spread_correct'].mean()
        overall_accuracy = df['overall_correct'].mean()
        
        self.logger.info("Test set summary (predictions CSV written)")
        self.logger.info(f"  samples: {total_samples}")
        self.logger.info(f"  fire_spread_acc: {fire_spread_accuracy:.4f} ({fire_spread_accuracy*100:.2f}%)")
        self.logger.info(f"  overall_acc: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        self.logger.debug(f"Predictions CSV saved to: {csv_path}")
        
        # 计算更多指标：Precision(宏), F1(宏), MSE, WMSE, Brier Score, RPS
        

        y_true = df['fire_spread_true'].to_numpy()
        y_pred = df['fire_spread_pred'].to_numpy()

        # 类别集合（按实际出现的标签）
        labels_present = sorted(list(set(y_true.tolist())))

        # 计算每类的 TP/FP/FN
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        for cls in labels_present:
            tp = int(((y_true == cls) & (y_pred == cls)).sum())
            fp = int(((y_true != cls) & (y_pred == cls)).sum())
            fn = int(((y_true == cls) & (y_pred != cls)).sum())
            precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_c = (2 * precision_c * recall_c / (precision_c + recall_c)) if (precision_c + recall_c) > 0 else 0.0
            precision_per_class.append(precision_c)
            recall_per_class.append(recall_c)
            f1_per_class.append(f1_c)

        precision_macro = float(np.mean(precision_per_class)) if precision_per_class else 0.0
        f1_macro = float(np.mean(f1_per_class)) if f1_per_class else 0.0

        # MSE（基于类别索引的平方误差）
        mse = float(np.mean((y_true - y_pred) ** 2)) if total_samples > 0 else 0.0

        # WMSE（按真实类别的反频率加权，并归一到样本数）
        class_counts = df['fire_spread_true'].value_counts().to_dict()
        weights = np.array([1.0 / class_counts.get(int(t), 1) for t in y_true], dtype=float)
        # 归一化到权重和等于样本数，使其与 MSE 可比
        if weights.sum() > 0:
            weights = weights * (len(weights) / weights.sum())
        wmse = float(np.sum(weights * ((y_true - y_pred) ** 2)) / len(weights)) if len(weights) > 0 else 0.0

        # Brier Score（多分类）：对每个样本 sum_k (p_k - y_k)^2 的平均
        class_order = [2, 3, 4]
        prob_cols = {k: f'fire_spread_prob_{k}' for k in class_order}
        # 若缺列，填0
        for col in prob_cols.values():
            if col not in df.columns:
                df[col] = 0.0
        probs = df[[prob_cols[k] for k in class_order]].to_numpy(dtype=float)
        # 观测 one-hot
        one_hot = np.zeros_like(probs)
        # 将真实类别映射到索引
        cls_to_idx = {c: i for i, c in enumerate(class_order)}
        for i, t in enumerate(y_true):
            if int(t) in cls_to_idx:
                one_hot[i, cls_to_idx[int(t)]] = 1.0
        brier_per_sample = np.sum((probs - one_hot) ** 2, axis=1)
        brier_score = float(np.mean(brier_per_sample)) if len(brier_per_sample) > 0 else 0.0

        # RPS（Ranked Probability Score，序有序多分类）：使用累计概率，常用 K-1 个阈值
        # 计算每个样本在阈值 k ∈ class_order[:-1] 上的累计预测与累计观测
        cum_probs = np.cumsum(probs, axis=1)
        # 累计观测：在阈值处是否已达到真实类别（真实类别 <= 阈值则为1）
        cum_obs = np.zeros_like(cum_probs)
        for j, k in enumerate(class_order):
            mask = (y_true <= k)
            cum_obs[:, j] = mask.astype(float)
        # 仅使用前 K-1 项
        rps_terms = (cum_probs[:, :-1] - cum_obs[:, :-1]) ** 2
        rps = float(np.mean(np.sum(rps_terms, axis=1))) if rps_terms.size > 0 else 0.0

        # 保存汇总统计（包含每个类别的准确率）
        summary_path = f"./output/predictions_summary_{timestamp}.csv"
        summary_data = [
            ['total_samples', total_samples, 100],
            ['fire_spread_accuracy', fire_spread_accuracy, fire_spread_accuracy*100],
            ['overall_accuracy', overall_accuracy, overall_accuracy*100],
            ['precision_macro', precision_macro, precision_macro*100],
            ['f1_macro', f1_macro, f1_macro*100],
            ['mse', mse, None],
            ['wmse', wmse, None],
            ['brier_score', brier_score, None],
            ['rps', rps, None]
        ]
        
        # 添加每个类别的准确率（recall）
        for i, cls in enumerate(labels_present):
            recall = recall_per_class[i]
            summary_data.append([f'class_{cls}_accuracy', recall, recall*100])
        
        # 添加每个类别的精确率
        for i, cls in enumerate(labels_present):
            precision = precision_per_class[i]
            summary_data.append([f'class_{cls}_precision', precision, precision*100])
        
        # 添加每个类别的F1分数
        for i, cls in enumerate(labels_present):
            f1 = f1_per_class[i]
            summary_data.append([f'class_{cls}_f1', f1, f1*100])
        
        # 添加每个类别的样本数
        for cls in labels_present:
            count = int((y_true == cls).sum())
            summary_data.append([f'class_{cls}_samples', count, count/total_samples*100])
        
        summary_df = pd.DataFrame(summary_data, columns=['metric', 'value', 'percentage'])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        
        self.logger.debug(f"Summary CSV saved to: {summary_path}")
        
    def save_training_summary(self, 
                            train_data: List[FireIncidentData],
                            val_data: List[FireIncidentData],
                            test_data: List[FireIncidentData],
                            metrics: Dict,
                            save_path: str = "./output/training_summary.json",
                            model_path: str = "./models/firellm_finetuned"):
        """保存训练总结"""
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        summary = {
            'model_info': {
                'model_name': self.model_name,
                'training_date': datetime.now().isoformat(),
                'model_path': model_path
            },
            'data_info': {
                'total_samples': len(train_data) + len(val_data) + len(test_data),
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'test_samples': len(test_data)
            },
            'training_config': self.training_config,
            'performance_metrics': metrics
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Training summary saved to {save_path}")
        
    def run_complete_training_pipeline(self, 
                                     data_path: str,
                                     filter_baltimore: bool = True,
                                     save_path: str = "./models/firellm_finetuned",
                                     **kwargs):
        """运行完整的训练流程"""
        
        self.logger.info("Starting complete training pipeline...")
        
        try:
            # 设置wandb
            self.setup_wandb()
            
            # 加载和准备数据
            train_data, val_data, test_data = self.load_and_prepare_data(
                data_path, 
                filter_baltimore=filter_baltimore
            )
            
            # 初始化模型
            self.initialize_model()
            
            # 训练模型
            self.train_model(train_data, val_data, save_path, **kwargs)
            
            # 评估模型
            metrics = self.evaluate_model(test_data)
            
            # 保存训练总结
            self.save_training_summary(train_data, val_data, test_data, metrics)
            
            self.logger.info("Training pipeline completed successfully!")
            
            return {
                'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data,
                'metrics': metrics,
                'model_path': save_path
            }
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {str(e)}")
            raise
        finally:
            if self.use_wandb:
                wandb.finish()
                
    def load_trained_model(self, model_path: str):
        """加载训练好的模型"""
        
        self.logger.info(f"Loading trained model from {model_path}")
        
        if self.model is None:
            self.initialize_model()
            
        # 加载LoRA适配器权重
        if os.path.exists(f"{model_path}/adapter_model.bin"):
            # 检查模型是否已经是 PeftModel
            from peft import PeftModel
            import peft
            
            if isinstance(self.model.model, PeftModel):
                # 如果已经是 PeftModel，需要完全移除adapter后重新加载
                self.logger.info("Model is already a PeftModel, removing adapter and reloading...")
                
                # 方法：使用 unload() 完全移除 PEFT wrapper
                try:
                    # unload() 会返回原始的base model（不带任何adapter）
                    base_model = self.model.model.unload()
                    self.logger.info("Successfully unloaded existing adapter")
                except Exception as e:
                    self.logger.warning(f"unload() failed: {e}, using get_base_model() instead")
                    base_model = self.model.model.get_base_model()
                
                # 在干净的base model上加载训练好的adapter
                self.model.model = PeftModel.from_pretrained(
                    base_model,
                    model_path,
                    is_trainable=False  # 测试模式，不需要训练
                )
                self.logger.info("LoRA adapter loaded successfully on clean base model")
            else:
                # 如果不是 PeftModel，正常加载
                self.model.model = PeftModel.from_pretrained(
                    self.model.model,
                    model_path,
                    is_trainable=False  # 测试模式，不需要训练
                )
                self.logger.info("LoRA adapter loaded successfully")
            
            # 强制将所有参数移动到正确的设备上
            self.model.model = self.model.model.to(self.model.device)
            # 确保LoRA权重也在正确的设备上
            for name, param in self.model.model.named_parameters():
                if param.device != torch.device(self.model.device):
                    param.data = param.data.to(self.model.device)
            
            # 设置为评估模式
            self.model.model.eval()
            self.logger.info("Model set to evaluation mode")
            
            # 加载分类头权重
            if os.path.exists(f"{model_path}/classifier_head.bin"):
                classifier_head_path = f"{model_path}/classifier_head.bin"
                classifier_head_state = torch.load(classifier_head_path, map_location=self.model.device)
                self.model.classifier.load_state_dict(classifier_head_state)
                self.logger.info("Classifier head weights loaded successfully")
            else:
                self.logger.warning("Classifier head weights not found, using randomly initialized weights")
        else:
            # 尝试加载完整模型
            try:
                self.model.model = torch.load(f"{model_path}/pytorch_model.bin", map_location=self.model.device)
                self.logger.info("Full model loaded successfully")
            except FileNotFoundError:
                raise FileNotFoundError(f"Neither LoRA adapter nor full model found in {model_path}")
        
        # 加载tokenizer
        self.model.tokenizer = self.model.tokenizer.from_pretrained(model_path)
        
        self.logger.info("Trained model loaded successfully")
        
    def predict_single_incident(self, incident_data: FireIncidentData) -> Dict:
        """预测单个火灾事件"""
        
        if self.model is None:
            raise ValueError("Model not initialized. Please call initialize_model() first.")
            
        prediction = self.model.predict(incident_data)
        
        self.logger.info(f"Prediction for incident {incident_data.incident_key}:")
        self.logger.info(f"  Fire spread: {prediction['prediction']['fire_spread']['class_name']}")
        self.logger.info(f"  Casualty: {prediction['prediction']['casualty']['class_name']}")
        
        return prediction

    def load_diverse_training_data_with_alignment(self, 
                                                diverse_data_path: str,
                                                original_data_path: str = None) -> tuple:
        """加载多样化训练数据，确保与原始数据划分对齐，并包含完整原始数据"""
        
        self.logger.info(f"Loading diverse training data from {diverse_data_path}")
        
        # 读取多样化数据
        with open(diverse_data_path, 'r', encoding='utf-8') as f:
            diverse_data = json.load(f)
        
        self.logger.info(f"Loaded {len(diverse_data)} diverse training samples")
        
        # 如果提供了原始数据路径，加载完整原始数据并与rephrase数据结合
        if original_data_path and os.path.exists(original_data_path):
            self.logger.info(f"Loading complete original data from {original_data_path}")
            train_data, val_data, test_data = self._load_complete_data_with_alignment(
                diverse_data, original_data_path
            )
        else:
            self.logger.info("Using new split for diverse data only")
            # 转换为训练格式
            train_samples = []
            for sample in diverse_data:
                train_sample = {
                    'prompt': sample['training_prompt'],
                    'fire_spread_label': sample['fire_spread_label'],
                    'incident_key': sample['incident_key'],
                    'is_rephrased': sample.get('is_rephrased', False)
                }
                train_samples.append(train_sample)
            
            # 分割数据
            train_data, val_data, test_data = self._split_diverse_data(train_samples)
        
        self.logger.info(f"Complete data prepared: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _load_complete_data_with_alignment(self, diverse_data: List[Dict], original_data_path: str) -> tuple:
        """加载完整原始数据并与rephrase数据结合，确保对齐"""
        
        self.logger.info("Loading complete original data and combining with rephrase data...")
        
        # 加载完整原始数据
        original_incident_data = self._load_complete_original_data(original_data_path)
        
        # 获取原始数据的划分信息
        original_split_info = self._load_original_split_info(original_data_path)
        
        # 创建incident_key到划分的映射
        split_mapping = {}
        for split_name, incident_keys in original_split_info.items():
            for incident_key in incident_keys:
                split_mapping[incident_key] = split_name
        
        # 按原始划分组织数据
        train_data = []
        val_data = []
        test_data = []
        
        # 首先添加所有原始数据
        for incident in original_incident_data:
            incident_key = incident.incident_key
            split_name = split_mapping.get(incident_key, 'train')
            
            # 创建原始数据的prompt
            original_prompt = self.model.create_fire_prompt(incident) if self.model else f"Original prompt for {incident_key}"
            
            train_sample = {
                'prompt': original_prompt,
                'fire_spread_label': incident.fire_spread_label,
                'incident_key': incident_key,
                'is_rephrased': False,
                'original_split': split_name
            }
            
            if split_name == 'train':
                train_data.append(train_sample)
            elif split_name == 'val':
                val_data.append(train_sample)
            elif split_name == 'test':
                test_data.append(train_sample)
            else:
                train_data.append(train_sample)
        
        # 然后添加rephrase数据，确保对齐
        for sample in diverse_data:
            incident_key = sample['incident_key']
            split_name = split_mapping.get(incident_key, 'train')
            
            train_sample = {
                'prompt': sample['training_prompt'],
                'fire_spread_label': sample['fire_spread_label'],
                'incident_key': incident_key,
                'is_rephrased': True,
                'original_split': split_name
            }
            
            if split_name == 'train':
                train_data.append(train_sample)
            elif split_name == 'val':
                val_data.append(train_sample)
            elif split_name == 'test':
                test_data.append(train_sample)
            else:
                train_data.append(train_sample)
        
        # 验证对齐结果
        self._validate_complete_alignment(train_data, val_data, test_data, original_split_info)
        
        return train_data, val_data, test_data
    
    def _load_complete_original_data(self, original_data_path: str) -> List[FireIncidentData]:
        """加载完整的原始数据"""
        
        self.logger.info(f"Loading complete original data from {original_data_path}")
        
        # 使用data_processor加载完整数据
        processed_data = self.data_processor.load_data(original_data_path)
        
        # 筛选Baltimore数据
        processed_data = self.data_processor.filter_baltimore_data(processed_data)
        
        # 数据清洗
        processed_data = self.data_processor.preprocess_data(processed_data)
        
        # 转换为FireIncidentData对象
        incident_data = self.data_processor.convert_to_incident_data(processed_data)
        
        self.logger.info(f"Loaded {len(incident_data)} complete original samples")
        
        return incident_data
    
    def _align_with_original_split(self, diverse_data: List[Dict], original_data_path: str) -> tuple:
        """将多样化数据与原始数据划分对齐"""
        
        # 读取原始数据的划分信息
        original_split_info = self._load_original_split_info(original_data_path)
        
        # 创建incident_key到划分的映射
        split_mapping = {}
        for split_name, incident_keys in original_split_info.items():
            for incident_key in incident_keys:
                split_mapping[incident_key] = split_name
        
        # 按原始划分组织多样化数据
        train_data = []
        val_data = []
        test_data = []
        
        for sample in diverse_data:
            incident_key = sample['incident_key']
            split_name = split_mapping.get(incident_key, 'train')  # 默认放入训练集
            
            train_sample = {
                'prompt': sample['training_prompt'],
                'fire_spread_label': sample['fire_spread_label'],
                'incident_key': incident_key,
                'is_rephrased': sample.get('is_rephrased', False),
                'original_split': split_name
            }
            
            if split_name == 'train':
                train_data.append(train_sample)
            elif split_name == 'val':
                val_data.append(train_sample)
            elif split_name == 'test':
                test_data.append(train_sample)
            else:
                # 未知的划分，默认放入训练集
                train_data.append(train_sample)
        
        # 验证对齐结果
        self._validate_alignment(train_data, val_data, test_data, original_split_info)
        
        return train_data, val_data, test_data
    
    def _load_original_split_info(self, original_data_path: str) -> Dict[str, List[str]]:
        """加载原始数据的划分信息"""
        
        # 检查是否是CSV文件（完整的原始数据）
        if original_data_path.endswith('.csv'):
            self.logger.info(f"Loading split info from CSV file: {original_data_path}")
            return self._load_split_info_from_csv(original_data_path)
        
        # 尝试从不同的原始数据文件加载划分信息
        possible_paths = [
            original_data_path,
            "./data/processed_prompts_training/fire_spread_class_3_5_training_prompts.json",
            "./data/processed_prompts_complete/fire_spread_class_3_5_complete_prompts.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        original_data = json.load(f)
                    
                    # 使用相同的分割逻辑
                    train_samples, val_samples, test_samples = self._split_original_data(original_data)
                    
                    # 提取incident_keys
                    train_keys = [item['incident_key'] for item in train_samples]
                    val_keys = [item['incident_key'] for item in val_samples]
                    test_keys = [item['incident_key'] for item in test_samples]
                    
                    self.logger.info(f"Loaded original split from {path}")
                    self.logger.info(f"Original split: Train={len(train_keys)}, Val={len(val_keys)}, Test={len(test_keys)}")
                    
                    return {
                        'train': train_keys,
                        'val': val_keys,
                        'test': test_keys
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load split info from {path}: {e}")
                    continue
        
        # 如果都失败了，返回空划分
        self.logger.warning("Could not load original split info, using new split")
        return {'train': [], 'val': [], 'test': []}
    
    def _load_split_info_from_csv(self, csv_path: str) -> Dict[str, List[str]]:
        """从CSV文件加载划分信息"""
        
        import pandas as pd
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            self.logger.info(f"Loaded CSV with {len(df)} rows")
            
            # 检查FIRE_SPRD列
            if 'FIRE_SPRD' not in df.columns:
                self.logger.error("FIRE_SPRD column not found in CSV")
                return {'train': [], 'val': [], 'test': []}
            
            # 检查INCIDENT_KEY列
            if 'INCIDENT_KEY' not in df.columns:
                self.logger.error("INCIDENT_KEY column not found in CSV")
                return {'train': [], 'val': [], 'test': []}
            
            # 过滤有效数据（排除标题行等）
            df = df[df['FIRE_SPRD'].isin([2, 3, 4, 5])]
            self.logger.info(f"Filtered data with valid FIRE_SPRD: {len(df)} rows")
            
            # 统计各类别数量
            class_counts = df['FIRE_SPRD'].value_counts().sort_index()
            self.logger.info(f"Class distribution: {dict(class_counts)}")
            
            # 使用相同的分割逻辑
            train_samples, val_samples, test_samples = self._split_csv_data(df)
            
            # 提取incident_keys
            train_keys = train_samples['INCIDENT_KEY'].tolist()
            val_keys = val_samples['INCIDENT_KEY'].tolist()
            test_keys = test_samples['INCIDENT_KEY'].tolist()
            
            self.logger.info(f"CSV split: Train={len(train_keys)}, Val={len(val_keys)}, Test={len(test_keys)}")
            
            return {
                'train': train_keys,
                'val': val_keys,
                'test': test_keys
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load split info from CSV: {e}")
            return {'train': [], 'val': [], 'test': []}
    
    def _split_csv_data(self, df: pd.DataFrame) -> tuple:
        """使用与原始训练相同的逻辑分割CSV数据"""
        
        import random
        random.seed(42)
        
        # 按类别分别分割
        class_2_data = df[df['FIRE_SPRD'] == 2]
        class_3_data = df[df['FIRE_SPRD'] == 3]
        class_4_data = df[df['FIRE_SPRD'] == 4]
        class_5_data = df[df['FIRE_SPRD'] == 5]
        
        # 随机打乱
        class_2_data = class_2_data.sample(frac=1, random_state=42).reset_index(drop=True)
        class_3_data = class_3_data.sample(frac=1, random_state=42).reset_index(drop=True)
        class_4_data = class_4_data.sample(frac=1, random_state=42).reset_index(drop=True)
        class_5_data = class_5_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 计算分割点 (70% train, 15% val, 15% test)
        class_2_train_split = int(len(class_2_data) * 0.7)
        class_2_val_split = int(len(class_2_data) * 0.85)
        class_3_train_split = int(len(class_3_data) * 0.7)
        class_3_val_split = int(len(class_3_data) * 0.85)
        class_4_train_split = int(len(class_4_data) * 0.7)
        class_4_val_split = int(len(class_4_data) * 0.85)
        class_5_train_split = int(len(class_5_data) * 0.7)
        class_5_val_split = int(len(class_5_data) * 0.85)
        
        # 分割数据
        train_data = pd.concat([
            class_2_data[:class_2_train_split],
            class_3_data[:class_3_train_split],
            class_4_data[:class_4_train_split],
            class_5_data[:class_5_train_split]
        ])
        
        val_data = pd.concat([
            class_2_data[class_2_train_split:class_2_val_split],
            class_3_data[class_3_train_split:class_3_val_split],
            class_4_data[class_4_train_split:class_4_val_split],
            class_5_data[class_5_train_split:class_5_val_split]
        ])
        
        test_data = pd.concat([
            class_2_data[class_2_val_split:],
            class_3_data[class_3_val_split:],
            class_4_data[class_4_val_split:],
            class_5_data[class_5_val_split:]
        ])
        
        # 再次打乱
        train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
        val_data = val_data.sample(frac=1, random_state=42).reset_index(drop=True)
        test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return train_data, val_data, test_data
    
    def _split_original_data(self, original_data: List[Dict]) -> tuple:
        """使用与原始训练相同的逻辑分割数据"""
        
        import random
        random.seed(42)
        
        # 按类别分别分割
        class_3_data = [item for item in original_data if item['fire_spread_label'] == 3]
        class_5_data = [item for item in original_data if item['fire_spread_label'] == 5]
        
        # 随机打乱
        random.shuffle(class_3_data)
        random.shuffle(class_5_data)
        
        # 计算分割点 (70% train, 15% val, 15% test)
        class_3_train_split = int(len(class_3_data) * 0.7)
        class_3_val_split = int(len(class_3_data) * 0.85)
        class_5_train_split = int(len(class_5_data) * 0.7)
        class_5_val_split = int(len(class_5_data) * 0.85)
        
        # 分割数据
        train_data = class_3_data[:class_3_train_split] + class_5_data[:class_5_train_split]
        val_data = class_3_data[class_3_train_split:class_3_val_split] + class_5_data[class_5_train_split:class_5_val_split]
        test_data = class_3_data[class_3_val_split:] + class_5_data[class_5_val_split:]
        
        # 再次打乱
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        return train_data, val_data, test_data
    
    def _validate_alignment(self, train_data: List[Dict], val_data: List[Dict], 
                          test_data: List[Dict], original_split_info: Dict[str, List[str]]):
        """验证对齐结果"""
        
        # 统计各集合中的incident_keys
        train_keys = set(item['incident_key'] for item in train_data)
        val_keys = set(item['incident_key'] for item in val_data)
        test_keys = set(item['incident_key'] for item in test_data)
        
        # 检查是否有重复
        train_val_overlap = train_keys & val_keys
        train_test_overlap = train_keys & test_keys
        val_test_overlap = val_keys & test_keys
        
        if train_val_overlap:
            self.logger.warning(f"Found {len(train_val_overlap)} overlapping keys between train and val")
        if train_test_overlap:
            self.logger.warning(f"Found {len(train_test_overlap)} overlapping keys between train and test")
        if val_test_overlap:
            self.logger.warning(f"Found {len(val_test_overlap)} overlapping keys between val and test")
        
        # 检查与原始划分的一致性
        if original_split_info:
            original_train_keys = set(original_split_info.get('train', []))
            original_val_keys = set(original_split_info.get('val', []))
            original_test_keys = set(original_split_info.get('test', []))
            
            # 检查训练集一致性
            train_consistency = len(train_keys & original_train_keys) / len(original_train_keys) if original_train_keys else 0
            val_consistency = len(val_keys & original_val_keys) / len(original_val_keys) if original_val_keys else 0
            test_consistency = len(test_keys & original_test_keys) / len(original_test_keys) if original_test_keys else 0
            
            self.logger.info(f"Split consistency: Train={train_consistency:.2%}, Val={val_consistency:.2%}, Test={test_consistency:.2%}")
            
            if train_consistency < 0.95 or val_consistency < 0.95 or test_consistency < 0.95:
                self.logger.warning("Low consistency with original split detected!")
        
        # 打印统计信息
        self.logger.info(f"Final split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # 按类别统计（3类分类）
        for split_name, data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
            class_2_count = sum(1 for item in data if item['fire_spread_label'] == 2)
            class_3_count = sum(1 for item in data if item['fire_spread_label'] == 3)
            class_4_count = sum(1 for item in data if item['fire_spread_label'] == 4)
            self.logger.info(f"{split_name} class distribution: Class 2={class_2_count}, Class 3={class_3_count}, Class 4={class_4_count}")

    def load_diverse_training_data(self, diverse_data_path: str) -> tuple:
        """加载多样化训练数据（直接使用rephrase后的prompt）- 备用方法"""
        
        self.logger.info(f"Loading diverse training data from {diverse_data_path}")
        
        # 读取多样化数据
        with open(diverse_data_path, 'r', encoding='utf-8') as f:
            diverse_data = json.load(f)
        
        self.logger.info(f"Loaded {len(diverse_data)} diverse training samples")
        
        # 转换为训练格式
        train_samples = []
        for sample in diverse_data:
            train_sample = {
                'prompt': sample['training_prompt'],  # 使用rephrase后的prompt
                'fire_spread_label': sample['fire_spread_label'],
                'incident_key': sample['incident_key'],
                'is_rephrased': sample.get('is_rephrased', False)
            }
            train_samples.append(train_sample)
        
        # 分割数据
        train_data, val_data, test_data = self._split_diverse_data(train_samples)
        
        self.logger.info(f"Diverse data prepared: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _split_diverse_data(self, diverse_samples: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple:
        """分割多样化数据 - 备用方法"""
        
        import random
        random.seed(42)
        
        # 按类别分别分割（3类分类）
        class_2_data = [item for item in diverse_samples if item['fire_spread_label'] == 2]
        class_3_data = [item for item in diverse_samples if item['fire_spread_label'] == 3]
        class_4_data = [item for item in diverse_samples if item['fire_spread_label'] == 4]
        
        # 随机打乱
        random.shuffle(class_2_data)
        random.shuffle(class_3_data)
        random.shuffle(class_4_data)
        
        # 计算分割点
        class_2_train_split = int(len(class_2_data) * train_ratio)
        class_2_val_split = int(len(class_2_data) * (train_ratio + val_ratio))
        class_3_train_split = int(len(class_3_data) * train_ratio)
        class_3_val_split = int(len(class_3_data) * (train_ratio + val_ratio))
        class_4_train_split = int(len(class_4_data) * train_ratio)
        class_4_val_split = int(len(class_4_data) * (train_ratio + val_ratio))
        
        # 分割数据
        train_data = class_2_data[:class_2_train_split] + class_3_data[:class_3_train_split] + class_4_data[:class_4_train_split]
        val_data = class_2_data[class_2_train_split:class_2_val_split] + class_3_data[class_3_train_split:class_3_val_split] + class_4_data[class_4_train_split:class_4_val_split]
        test_data = class_2_data[class_2_val_split:] + class_3_data[class_3_val_split:] + class_4_data[class_4_val_split:]
        
        # 再次打乱
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)
        
        return train_data, val_data, test_data

    def train_model_with_diverse_data(self, train_data: List[Dict], val_data: List[Dict], 
                                    save_path: str, **kwargs):
        """使用多样化数据训练模型"""
        
        self.logger.info("开始使用多样化数据训练模型...")
        self.logger.info(f"模型将保存到: {save_path}")
        
        # 更新训练配置
        if kwargs:
            self.update_training_config(**kwargs)
        
        # 开始训练
        self._train_with_diverse_data(train_data, val_data, save_path)
        
        self.logger.info("多样化数据训练完成！")
    
    def _train_with_diverse_data(self, train_data: List[Dict], val_data: List[Dict], save_path: str):
        """使用多样化数据进行训练的内部方法"""
        
        if self.model is None:
            raise ValueError("Model not initialized. Please call initialize_model() first.")
        
        # 准备训练数据
        self.logger.info("准备训练数据...")
        
        # 转换为模型需要的格式
        train_dataset = []
        for sample in train_data:
            train_sample = self.model.create_training_data_from_prompt(
                sample['prompt'], 
                sample['fire_spread_label']
            )
            train_dataset.append(train_sample)
        
        val_dataset = []
        for sample in val_data:
            val_sample = self.model.create_training_data_from_prompt(
                sample['prompt'], 
                sample['fire_spread_label']
            )
            val_dataset.append(val_sample)
        
        self.logger.info(f"训练数据准备完成: Train={len(train_dataset)}, Val={len(val_dataset)}")
        
        # 设置优化器和调度器
        optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=self.training_config.get('learning_rate', 2e-4),
            weight_decay=self.training_config.get('weight_decay', 0.01)
        )
        
        total_steps = len(train_dataset) * self.training_config.get('num_epochs', 10)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.training_config.get('learning_rate', 2e-4),
            total_steps=total_steps,
            pct_start=0.1
        )
        
        # 训练循环
        self.logger.info("开始训练循环...")
        self.model.model.train()
        
        num_epochs = self.training_config.get('num_epochs', 10)
        batch_size = self.training_config.get('batch_size', 8)
        gradient_accumulation_steps = self.training_config.get('gradient_accumulation_steps', 4)
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = self.training_config.get('early_stopping_patience', 5)
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # 训练阶段
            train_loss = 0.0
            optimizer.zero_grad()
            
            for i, sample in enumerate(train_dataset):
                # 前向传播
                outputs = self.model.model(
                    input_ids=sample['input_ids'].unsqueeze(0).to(self.model.device),
                    attention_mask=sample['attention_mask'].unsqueeze(0).to(self.model.device)
                )
                
                # 计算损失
                logits = outputs.logits[0, -1, self.model.class_token_ids]
                loss = torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), 
                    torch.tensor([sample['class_label']]).to(self.model.device)
                )
                
                # 反向传播
                loss.backward()
                train_loss += loss.item()
                
                # 梯度累积
                if (i + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            avg_train_loss = train_loss / len(train_dataset)
            
            # 验证阶段
            self.model.model.eval()
            val_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                for sample in val_dataset:
                    outputs = self.model.model(
                        input_ids=sample['input_ids'].unsqueeze(0).to(self.model.device),
                        attention_mask=sample['attention_mask'].unsqueeze(0).to(self.model.device)
                    )
                    
                    logits = outputs.logits[0, -1, self.model.class_token_ids]
                    loss = torch.nn.functional.cross_entropy(
                        logits.unsqueeze(0), 
                        torch.tensor([sample['class_label']]).to(self.model.device)
                    )
                    
                    val_loss += loss.item()
                    
                    # 计算准确率
                    predicted_class = torch.argmax(logits).item()
                    if predicted_class == sample['class_label']:
                        correct_predictions += 1
                    total_predictions += 1
            
            avg_val_loss = val_loss / len(val_dataset)
            val_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            self.logger.info(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.4f}")
            
            # 记录到wandb
            if self.use_wandb and WANDB_AVAILABLE:
                try:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'val_accuracy': val_accuracy,
                        'learning_rate': scheduler.get_last_lr()[0]
                    })
                except Exception as e:
                    self.logger.warning(f"wandb.log failed: {e}")
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # 保存最佳模型
                os.makedirs(save_path, exist_ok=True)
                self.model.model.save_pretrained(save_path)
                self.model.tokenizer.save_pretrained(save_path)
                self.logger.info(f"最佳模型已保存到: {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"早停触发，在epoch {epoch + 1}停止训练")
                    break
            
            self.model.model.train()
        
        self.logger.info("训练完成！")
        
        # 最终保存
        final_save_path = f"{save_path}_final"
        os.makedirs(final_save_path, exist_ok=True)
        self.model.model.save_pretrained(final_save_path)
        self.model.tokenizer.save_pretrained(final_save_path)
        self.logger.info(f"最终模型已保存到: {final_save_path}")

