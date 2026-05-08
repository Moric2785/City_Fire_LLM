#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FireLLM-Classifier Training Script - Supports single configuration training and grid search
Modified to use Llama 3.1-8B-Instruct
"""
import os
import sys
import logging
import argparse
import json
import itertools
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fllm.trainer_classifier import FireLLMTrainer
from fllm.utils import setup_directories
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

# # ===== 修改：使用 Llama 3.1-8B-Instruct =====
# DEFAULT_DATA_PATH = os.getenv("FIRELLM_DATA_PATH", "./data/processed_data/433_wo_la_wash.csv")
# DEFAULT_MODEL_PATH = os.getenv("FIRELLM_MODEL_PATH", "./models/llama3.1-8b-instruct")
# DEFAULT_MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"  # 新增：模型仓库ID
# ============================================

# ===== 使用 DeepSeekR1-14B =====
DEFAULT_DATA_PATH = os.getenv("FIRELLM_DATA_PATH", "./data/processed_data/433_wo_la_wash.csv")
DEFAULT_MODEL_PATH = os.getenv("FIRELLM_MODEL_PATH","./models/DeepSeek-R1-Distill-Qwen-7B")
# DeepSeek R1 14B 模型仓库 ID
DEFAULT_MODEL_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# ============================================


def ensure_model_available(model_path: str = DEFAULT_MODEL_PATH,
                           repo_id: str = DEFAULT_MODEL_REPO,  # 修改：使用新的默认仓库
                           token_env_var: str = "HUGGINGFACE_HUB_TOKEN",
                           logger=None) -> bool:
    """Ensure the model exists locally; if not, attempt to download from Hugging Face.

    Returns True if model is available (existing or successfully downloaded), False otherwise.
    """
    if os.path.exists(model_path):
        # 检查是否包含必要的模型文件
        required_files = ['config.json']
        has_weights = any([
            os.path.exists(os.path.join(model_path, f)) 
            for f in ['model.safetensors', 'pytorch_model.bin', 'model-00001-of-00004.safetensors']
        ])
        
        if has_weights and all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
            if logger:
                logger.info(f"✅ Model already exists and is complete: {model_path}")
            return True
        else:
            if logger:
                logger.warning(f"⚠️ Model directory exists but incomplete: {model_path}")
                logger.info(f"🔄 Will attempt to re-download...")

    if snapshot_download is None:
        if logger:
            logger.error("❌ huggingface_hub is not installed. Please `pip install huggingface-hub`.")
        return False

    hf_token = os.getenv(token_env_var)
    if not hf_token:
        if logger:
            logger.error(f"❌ Hugging Face token not found in env var '{token_env_var}'.")
            logger.info(f"💡 Set it with: export {token_env_var}='your_token_here'")
            logger.info(f"💡 Or pass it via command line: --hf_token 'your_token_here'")
        return False

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)

    try:
        if logger:
            logger.info(f"📥 Downloading model from {repo_id} to {model_path} ...")
            logger.info(f"⏳ This may take several minutes (model size: ~16GB)...")
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            token=hf_token,
            resume_download=True,  # 支持断点续传
            max_workers=4  # 加速下载
        )
        
        if logger:
            logger.info(f"✅ Model downloaded successfully to: {model_path}")
        return True
    except Exception as e:
        if logger:
            logger.error(f"❌ Failed to download model: {e}")
            if "401" in str(e) or "403" in str(e):
                logger.error(f"🔐 Authentication failed. Please check:")
                logger.error(f"   1. Your token is valid")
                logger.error(f"   2. You have access to {repo_id}")
                logger.error(f"   3. Visit https://huggingface.co/{repo_id} to request access")
        return False

def single_training():
    """Single configuration training"""
    # Ensure log directory exists
    os.makedirs('./output', exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./output/training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    setup_directories()
    
    logger.info("🔥 Start FireLLM-Classifier single training (Llama 3.1-8B-Instruct)")
    
    try:
        # Configuration parameters - optimized hyperparameters for 3-class classification
        config = {
            'model_name': DEFAULT_MODEL_PATH,  # 修改：使用默认路径常量
            'data_path': DEFAULT_DATA_PATH,  
            'filter_baltimore': False, 
            'num_epochs': 5,  
            'batch_size': 5,  
            'learning_rate': 1e-4,  
            'use_4bit': True,
            'use_lora': True,
            'lora_r': 32,
            'lora_alpha': 64,
            'lora_dropout': 0.1,
            'gradient_accumulation_steps': 8,
            'weight_decay': 0.005,
            'warmup_ratio': 0.15,
            'min_lr': 5e-7,
            'label_smoothing': 0.05,
            'early_stopping_patience': 8,
            'lr_scheduler': 'cosine_with_restarts',
            'class_weights': False,
            'dropout': 0.15,
            'focal_loss': False,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'max_grad_norm': 1.0,
            'use_cls_head': True,
            'enable_data_balancing': False
        }
        
        logger.info(f"📋 Configuration: {json.dumps(config, indent=2)}")
        
        # Check if data file exists
        if not os.path.exists(config['data_path']):
            logger.error(f"❌ Data file not found: {config['data_path']}")
            logger.info("💡 Please ensure the data file exists")
            return
        
        # Ensure model is available locally (download from Hugging Face if missing)
        logger.info(f"🔍 Checking model availability: {config['model_name']}")
        if not ensure_model_available(config['model_name'], repo_id=DEFAULT_MODEL_REPO, logger=logger):
            logger.error(f"❌ Model not available: {config['model_name']}")
            logger.info("💡 Please set HUGGINGFACE_HUB_TOKEN and ensure you have access to the model")
            return
        
        # Initialize trainer
        trainer = FireLLMTrainer(
            model_name=config['model_name'],
            use_wandb=True,
            project_name="firellm-classifier"
        )
        
        # Setup wandb
        logger.info("📊 Initialize Weights & Biases...")
        trainer.setup_wandb()
        
        # Load and prepare data
        logger.info("📂 Load and prepare data...")
        train_data, val_data, test_data = trainer.load_and_prepare_data(
            data_path=config['data_path'],
            filter_baltimore=config['filter_baltimore'],
            save_processed=False
        )
        
        if config.get('enable_data_balancing', False):
            logger.info("⚖️ Apply balanced sampling on test set...")
            from collections import defaultdict
            import random
            
            test_by_class = defaultdict(list)
            for data in test_data:
                test_by_class[data.fire_spread_label].append(data)
            
            min_test_samples = min(len(test_by_class[cls]) for cls in [2,3,4])
            balanced_test_data = []
            for cls, ratio in [(2,4), (3,2), (4,2)]:
                target_count = min_test_samples * ratio
                if target_count > len(test_by_class[cls]):
                    target_count = len(test_by_class[cls])
                sampled = random.sample(test_by_class[cls], target_count)
                balanced_test_data.extend(sampled)
            
            random.shuffle(balanced_test_data)
            test_data = balanced_test_data
            logger.info(f"✅ Balanced test size: {len(test_data)}")
        else:
            logger.info(f"📊 Original test size: {len(test_data)}")
        
        # Initialize model
        logger.info("🤖 Initialize model...")
        trainer.initialize_model(
            use_4bit=config['use_4bit'],
            use_lora=config['use_lora'],
            lora_r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout']
        )
        
        # Train model
        logger.info("🚀 Start training...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"./models/firellm_llama31_lr{config['learning_rate']}_{timestamp}"
        
        logger.info(f"💾 Model will be saved to: {model_save_path}")
        
        trainer.train_model(
            train_data=train_data,
            val_data=val_data,
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            save_path=model_save_path,
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            weight_decay=config['weight_decay'],
            warmup_ratio=config['warmup_ratio'],
            min_lr=config['min_lr'],
            label_smoothing=config['label_smoothing'],
            early_stopping_patience=config['early_stopping_patience'],
            lr_scheduler=config['lr_scheduler'],
            class_weights=config['class_weights'],
            dropout=config['dropout'],
            focal_loss=config['focal_loss'],
            focal_alpha=config['focal_alpha'],
            focal_gamma=config['focal_gamma'],
            max_grad_norm=config['max_grad_norm'],
            use_cls_head=config['use_cls_head']
        )
        
        # Evaluate model
        logger.info("📈 Evaluate on test set...")
        metrics = trainer.evaluate_model(test_data)
        
        # Debug: Print metrics structure
        logger.info(f"📊 Metrics structure: {type(metrics)}")
        logger.info(f"📊 Metrics keys: {list(metrics.keys()) if isinstance(metrics, dict) else 'Not a dict'}")
        
        # Save training summary
        logger.info("💾 Save training summary...")
        trainer.save_training_summary(train_data, val_data, test_data, metrics, model_path=model_save_path)
        
        logger.info("✅ Training finished")
        logger.info(f"📁 Model saved to: {model_save_path}")
        logger.info("📁 Evaluation artifacts saved under ./output/")
        
        # Print final metrics (matching metrics_calculator return format - nested structure)
        logger.info("📊 Final metrics:")
        try:
            if 'fire_spread' in metrics and 'accuracy' in metrics['fire_spread']:
                logger.info(f"  🎯 fire_spread_acc: {metrics['fire_spread']['accuracy']:.4f}")
            elif 'fire_spread_accuracy' in metrics:
                # Fallback to flat structure
                logger.info(f"  🎯 fire_spread_acc: {metrics['fire_spread_accuracy']:.4f}")
            
            if 'overall' in metrics and 'accuracy' in metrics['overall']:
                logger.info(f"  🎯 overall_acc: {metrics['overall']['accuracy']:.4f}")
            elif 'overall_accuracy' in metrics:
                # Fallback to flat structure
                logger.info(f"  🎯 overall_acc: {metrics['overall_accuracy']:.4f}")
            
            logger.info(f"  📈 Total samples: {len(test_data)}")
        except Exception as e:
            logger.error(f"  ❌ Error printing metrics: {e}")
            logger.info(f"  📋 Metrics structure: {metrics}")
            raise
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def multi_training():
    """Multi-dataset training"""
    os.makedirs('./output', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./output/training_multi.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    setup_directories()
    
    # 确保模型可用
    if not ensure_model_available(DEFAULT_MODEL_PATH, repo_id=DEFAULT_MODEL_REPO, logger=logger):
        logger.error(f"❌ Model not available: {DEFAULT_MODEL_PATH}")
        return
    
    datasets = [
        "./data/processed_data/523.csv",
        "./data/processed_data/433.csv",
        "./data/processed_data/111.csv",
    ]
    for data_path in datasets:
        try:
            tmp_path = None
            try:
                df = pd.read_csv(data_path, low_memory=False)
                label_col = 'FIRE_SPRD' if 'FIRE_SPRD' in df.columns else ('fire_spread' if 'fire_spread' in df.columns else None)
                if label_col is not None:
                    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
                    df[label_col] = df[label_col].replace({5: 4, 45: 4})
                    df = df[df[label_col].isin([2, 3, 4])].copy()
                    stem = Path(data_path).stem
                    tmp_path = f"./processed_data/{stem}_mapped.csv"
                    df.to_csv(tmp_path, index=False)
                    data_path_to_use = tmp_path
                else:
                    data_path_to_use = data_path
            except Exception as _e:
                data_path_to_use = data_path
            
            config = {
                'model_name': DEFAULT_MODEL_PATH,  # 修改：使用默认路径
                'data_path': data_path_to_use,
                'filter_baltimore': False,
                'num_epochs': 5,
                'batch_size': 32,
                'learning_rate': 1e-4,
                'use_4bit': True,
                'use_lora': True,
                'lora_r': 32,
                'lora_alpha': 64,
                'lora_dropout': 0.1,
                'gradient_accumulation_steps': 8,
                'weight_decay': 0.005,
                'warmup_ratio': 0.15,
                'min_lr': 5e-7,
                'label_smoothing': 0.05,
                'early_stopping_patience': 8,
                'lr_scheduler': 'cosine_with_restarts',
                'class_weights': False,
                'dropout': 0.15,
                'focal_loss': False,
                'focal_alpha': 0.25,
                'focal_gamma': 2.0,
                'max_grad_norm': 1.0,
                'use_cls_head': True,
                'enable_data_balancing': False
            }
            if not os.path.exists(config['data_path']):
                logger.error(f"Data file not found: {config['data_path']}")
                continue
            
            trainer = FireLLMTrainer(
                model_name=config['model_name'],
                use_wandb=True,
                project_name="firellm-classifier"
            )
            trainer.setup_wandb()
            train_data, val_data, test_data = trainer.load_and_prepare_data(
                data_path=config['data_path'],
                filter_baltimore=config['filter_baltimore'],
                save_processed=False
            )
            trainer.initialize_model(
                use_4bit=config['use_4bit'],
                use_lora=config['use_lora'],
                lora_r=config['lora_r'],
                lora_alpha=config['lora_alpha'],
                lora_dropout=config['lora_dropout']
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = Path(config['data_path']).stem
            model_save_path = f"./models/firellm_llama31_lr{config['learning_rate']}_{stem}_{timestamp}"
            trainer.train_model(
                train_data=train_data,
                val_data=val_data,
                num_epochs=config['num_epochs'],
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate'],
                save_path=model_save_path,
                gradient_accumulation_steps=config['gradient_accumulation_steps'],
                weight_decay=config['weight_decay'],
                warmup_ratio=config['warmup_ratio'],
                min_lr=config['min_lr'],
                label_smoothing=config['label_smoothing'],
                early_stopping_patience=config['early_stopping_patience'],
                lr_scheduler=config['lr_scheduler'],
                class_weights=config['class_weights'],
                dropout=config['dropout'],
                focal_loss=config['focal_loss'],
                focal_alpha=config['focal_alpha'],
                focal_gamma=config['focal_gamma'],
                max_grad_norm=config['max_grad_norm'],
                use_cls_head=config['use_cls_head']
            )
            metrics = trainer.evaluate_model(test_data)
            trainer.save_training_summary(train_data, val_data, test_data, metrics, model_path=model_save_path)
        except Exception as e:
            logging.getLogger(__name__).error(str(e))
            raise

def lr_search():
    """Learning rate search"""
    os.makedirs('./output', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./output/lr_search.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    setup_directories()
    
    logger.info("🔍 Start LR search (Llama 3.1-8B-Instruct)")
    
    # 确保模型可用
    if not ensure_model_available(DEFAULT_MODEL_PATH, repo_id=DEFAULT_MODEL_REPO, logger=logger):
        logger.error(f"❌ Model not available: {DEFAULT_MODEL_PATH}")
        return False
    
    lr_groups = {
        'fine_core': [8e-5, 9e-5, 1.0e-4, 1.1e-4, 1.2e-4],
        'fine_edge': [7e-5, 1.3e-4, 1.5e-4]
    }
    
    base_config = {
        'model_name': DEFAULT_MODEL_PATH,  # 修改：使用默认路径
        'data_path': DEFAULT_DATA_PATH,
        'filter_baltimore': False,
        'num_epochs': 5,
        'batch_size': 16,
        'use_4bit': True,
        'use_lora': True,
        'lora_r': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.1,
        'gradient_accumulation_steps': 8,
        'weight_decay': 0.005,
        'warmup_ratio': 0.15,
        'min_lr': 5e-7,
        'label_smoothing': 0.05,
        'early_stopping_patience': 3,
        'lr_scheduler': 'cosine_with_restarts',
        'class_weights': False,
        'dropout': 0.15,
        'focal_loss': False,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'max_grad_norm': 1.0,
        'use_cls_head': True,
        'enable_data_balancing': False
    }
    
    results = []
    
    if not os.path.exists(base_config['data_path']):
        logger.error(f"Data file not found: {base_config['data_path']}")
        return False
    
    total_tests = sum(len(lrs) for lrs in lr_groups.values())
    logger.info(f"📊 Total {total_tests} learning rates to test")
    
    test_count = 0
    for group_name, lr_list in lr_groups.items():
        logger.info(f"\n🧪 Testing {group_name}: {lr_list}")
        
        for lr in lr_list:
            test_count += 1
            logger.info(f"🔬 Testing {test_count}/{total_tests}: LR={lr:.2e} ({group_name})")
            
            try:
                trainer = FireLLMTrainer(
                    model_name=base_config['model_name'],
                    use_wandb=False,
                    project_name="firellm-lrsearch"
                )
                
                train_data, val_data, test_data = trainer.load_and_prepare_data(
                    data_path=base_config['data_path'],
                    filter_baltimore=base_config['filter_baltimore'],
                    save_processed=False
                )
                
                trainer.initialize_model(
                    use_4bit=base_config['use_4bit'],
                    use_lora=base_config['use_lora'],
                    lora_r=base_config['lora_r'],
                    lora_alpha=base_config['lora_alpha'],
                    lora_dropout=base_config['lora_dropout']
                )
                
                start_time = datetime.now()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                trainer.train_model(
                    train_data=train_data,
                    val_data=val_data,
                    save_path=f"./models/lrsearch_llama31_{test_count}_{timestamp}",
                    learning_rate=lr,
                    num_epochs=base_config['num_epochs'],
                    batch_size=base_config['batch_size'],
                    gradient_accumulation_steps=base_config['gradient_accumulation_steps'],
                    weight_decay=base_config['weight_decay'],
                    warmup_ratio=base_config['warmup_ratio'],
                    min_lr=base_config['min_lr'],
                    label_smoothing=base_config['label_smoothing'],
                    early_stopping_patience=base_config['early_stopping_patience'],
                    lr_scheduler=base_config['lr_scheduler'],
                    class_weights=base_config['class_weights'],
                    dropout=base_config['dropout'],
                    focal_loss=base_config['focal_loss'],
                    focal_alpha=base_config['focal_alpha'],
                    focal_gamma=base_config['focal_gamma'],
                    max_grad_norm=base_config['max_grad_norm'],
                    use_cls_head=base_config['use_cls_head']
                )
                
                metrics = trainer.evaluate_model(test_data)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                result = {
                    'test_id': test_count,
                    'group': group_name,
                    'learning_rate': lr,
                    'metrics': metrics,
                    'duration_seconds': duration,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
                
                # Safe access with fallback
                try:
                    if 'overall' in metrics and 'accuracy' in metrics['overall']:
                        accuracy = metrics['overall']['accuracy']
                    elif 'overall_accuracy' in metrics:
                        accuracy = metrics['overall_accuracy']
                    else:
                        accuracy = 0
                except:
                    accuracy = 0
                
                logger.info(f"✅ LR={lr:.2e} completed: accuracy={accuracy:.4f}, duration={duration:.1f}s")
                
                with open('./output/lr_search_results.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                logger.error(f"❌ LR={lr:.2e} failed: {str(e)}")
                result = {
                    'test_id': test_count,
                    'group': group_name,
                    'learning_rate': lr,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
    
    logger.info("\n📊 Learning rate search results:")
    logger.info("=" * 80)
    
    successful_results = [r for r in results if 'metrics' in r]
    if successful_results:
        # Safe sort with fallback
        def get_accuracy(result):
            metrics = result['metrics']
            if 'overall' in metrics and 'accuracy' in metrics['overall']:
                return metrics['overall']['accuracy']
            elif 'overall_accuracy' in metrics:
                return metrics['overall_accuracy']
            return 0
        
        successful_results.sort(key=get_accuracy, reverse=True)
        
        logger.info("🏆 Best learning rate per group:")
        for group_name in lr_groups.keys():
            group_results = [r for r in successful_results if r['group'] == group_name]
            if group_results:
                best_result = group_results[0]
                accuracy = get_accuracy(best_result)
                lr = best_result['learning_rate']
                logger.info(f"  {group_name}: LR={lr:.2e}, accuracy={accuracy:.4f}")
        
        logger.info("\n🥇 Global best learning rate:")
        best_result = successful_results[0]
        best_lr = best_result['learning_rate']
        best_accuracy = get_accuracy(best_result)
        best_group = best_result['group']
        logger.info(f"  Learning rate: {best_lr:.2e}")
        logger.info(f"  Accuracy: {best_accuracy:.4f}")
        logger.info(f"  Group: {best_group}")
        
        best_config = {
            'best_learning_rate': best_lr,
            'best_accuracy': best_accuracy,
            'best_group': best_group,
            'all_results': results
        }
        with open('./output/best_lr_config.json', 'w', encoding='utf-8') as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Best config saved to: ./output/best_lr_config.json")
    else:
        logger.error("❌ No successful training results")
    
    logger.info(f"\n📁 Complete results saved to: ./output/lr_search_results.json")
    return len(successful_results) > 0

def test_mode(model_path=None, data_path=None):
    """Test mode - test trained model on new dataset"""
    os.makedirs('./output', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./output/test_mode.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    setup_directories()
    
    logger.info("🧪 Start FireLLM-Classifier test mode (Llama 3.1)")
    
    config = {
        'model_path': model_path or "./models/lrsearch_llama31_1",
        'data_path': data_path or DEFAULT_DATA_PATH,
        'filter_baltimore': False,
        'use_4bit': True,
        'use_lora': True,
        'lora_r': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.1,
        'use_cls_head': True
    }
    
    logger.info(f"📂 LoRA Adapter path: {config['model_path']}")
    logger.info(f"📂 Test data path: {config['data_path']}")
    logger.info(f"📂 Base model path: {DEFAULT_MODEL_PATH}")
    
    if not os.path.exists(config['model_path']):
        logger.error(f"❌ Model not found: {config['model_path']}")
        return
    
    if not os.path.exists(config['data_path']):
        logger.error(f"❌ Data file not found: {config['data_path']}")
        return
    
    # 确保基础模型可用
    if not ensure_model_available(DEFAULT_MODEL_PATH, repo_id=DEFAULT_MODEL_REPO, logger=logger):
        logger.error(f"❌ Base model not available: {DEFAULT_MODEL_PATH}")
        return
    
    try:
        trainer = FireLLMTrainer(
            model_name=DEFAULT_MODEL_PATH,  # 修改：使用默认路径
            use_wandb=False,
            project_name="firellm-test"
        )
        
        logger.info("📊 Loading test data...")
        train_data, val_data, test_data = trainer.load_and_prepare_data(
            data_path=config['data_path'],
            filter_baltimore=config['filter_baltimore'],
            save_processed=False
        )
        
        logger.info(f"✅ Data loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        trainer.initialize_model(
            use_4bit=config['use_4bit'],
            use_lora=config['use_lora'],
            lora_r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout']
        )
        
        logger.info(f"🔥 Loading trained model: {config['model_path']}")
        trainer.load_trained_model(config['model_path'])
        
        logger.info("📈 Evaluating model on test set...")
        metrics = trainer.evaluate_model(test_data)
        
        logger.info("\n" + "="*60)
        logger.info("📊 Test Results:")
        logger.info("="*60)
        
        # Safe access with fallback
        try:
            if 'fire_spread' in metrics and 'accuracy' in metrics['fire_spread']:
                fire_spread_acc = metrics['fire_spread']['accuracy']
            elif 'fire_spread_accuracy' in metrics:
                fire_spread_acc = metrics['fire_spread_accuracy']
            else:
                fire_spread_acc = 0
                
            if 'overall' in metrics and 'accuracy' in metrics['overall']:
                overall_acc = metrics['overall']['accuracy']
            elif 'overall_accuracy' in metrics:
                overall_acc = metrics['overall_accuracy']
            else:
                overall_acc = 0
                
            logger.info(f"  🎯 Fire Spread Accuracy: {fire_spread_acc:.4f} ({fire_spread_acc*100:.2f}%)")
            logger.info(f"  🎯 Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
        except Exception as e:
            logger.error(f"  ❌ Error accessing metrics: {e}")
            logger.info(f"  📋 Metrics structure: {metrics}")
        
        logger.info("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_summary = {
            'model_path': config['model_path'],
            'data_path': config['data_path'],
            'test_date': datetime.now().isoformat(),
            'test_samples': len(test_data),
            'metrics': metrics,
            'config': config
        }
        
        summary_path = f"./output/test_summary_{timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(test_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Test summary saved to: {summary_path}")
        
        # List all saved files
        logger.info("\n📁 Saved result files:")
        logger.info(f"  1. Test summary JSON: {summary_path}")
        logger.info(f"  2. Detailed predictions CSV: ./output/predictions_detailed_{timestamp}.csv")
        logger.info(f"  3. Summary statistics CSV: ./output/predictions_summary_{timestamp}.csv")
        logger.info(f"  4. Confusion matrices: ./output/confusion_matrices.png")
        logger.info(f"  5. Detailed report: ./output/detailed_report.txt")
        logger.info(f"  6. Metrics JSON: ./output/metrics.json")
        logger.info("✅ Test completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def quick_lr_test():
    """Quick learning rate test"""
    os.makedirs('./output', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./output/quick_lr_test.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    setup_directories()
    
    logger.info("⚡ Start FireLLM-Classifier quick learning rate test (Llama 3.1)")
    
    # Ensure model is available
    if not ensure_model_available(DEFAULT_MODEL_PATH, repo_id=DEFAULT_MODEL_REPO, logger=logger):
        logger.error(f"❌ Model not available: {DEFAULT_MODEL_PATH}")
        return False
    
    lr_candidates = [8e-5, 9e-5, 1.0e-4, 1.1e-4, 1.2e-4]
    
    base_config = {
        'model_name': DEFAULT_MODEL_PATH,
        'data_path': DEFAULT_DATA_PATH,
        'filter_baltimore': False,
        'num_epochs': 3,
        'batch_size': 16,
        'use_4bit': True,
        'use_lora': True,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'gradient_accumulation_steps': 4,
        'weight_decay': 0.005,
        'warmup_ratio': 0.1,
        'min_lr': 5e-7,
        'label_smoothing': 0.0,
        'early_stopping_patience': 2,
        'lr_scheduler': 'cosine',
        'class_weights': False,
        'dropout': 0.1,
        'focal_loss': False,
        'max_grad_norm': 1.0,
        'use_cls_head': True,
        'enable_data_balancing': False
    }
    
    results = []
    
    if not os.path.exists(base_config['data_path']):
        logger.error(f"Data file not found: {base_config['data_path']}")
        return False
    
    logger.info(f"📊 Total {len(lr_candidates)} learning rates to test")
    
    for i, lr in enumerate(lr_candidates):
        logger.info(f"🔬 Testing {i+1}/{len(lr_candidates)}: LR={lr:.2e}")
        
        try:
            trainer = FireLLMTrainer(
                model_name=base_config['model_name'],
                use_wandb=False,
                project_name="firellm-quicklr"
            )
            
            train_data, val_data, test_data = trainer.load_and_prepare_data(
                data_path=base_config['data_path'],
                filter_baltimore=base_config['filter_baltimore'],
                save_processed=False
            )
            
            trainer.initialize_model(
                use_4bit=base_config['use_4bit'],
                use_lora=base_config['use_lora'],
                lora_r=base_config['lora_r'],
                lora_alpha=base_config['lora_alpha'],
                lora_dropout=base_config['lora_dropout']
            )
            
            start_time = datetime.now()
            
            trainer.train_model(
                train_data=train_data,
                val_data=val_data,
                save_path=f"./models/quicklr_llama31_{i+1}",
                learning_rate=lr,
                num_epochs=base_config['num_epochs'],
                batch_size=base_config['batch_size'],
                gradient_accumulation_steps=base_config['gradient_accumulation_steps'],
                weight_decay=base_config['weight_decay'],
                warmup_ratio=base_config['warmup_ratio'],
                min_lr=base_config['min_lr'],
                label_smoothing=base_config['label_smoothing'],
                early_stopping_patience=base_config['early_stopping_patience'],
                lr_scheduler=base_config['lr_scheduler'],
                class_weights=base_config['class_weights'],
                dropout=base_config['dropout'],
                focal_loss=base_config['focal_loss'],
                max_grad_norm=base_config['max_grad_norm'],
                use_cls_head=base_config['use_cls_head']
            )
            
            metrics = trainer.evaluate_model(test_data)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'test_id': i + 1,
                'learning_rate': lr,
                'metrics': metrics,
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            # Safe access
            try:
                if 'overall' in metrics and 'accuracy' in metrics['overall']:
                    accuracy = metrics['overall']['accuracy']
                elif 'overall_accuracy' in metrics:
                    accuracy = metrics['overall_accuracy']
                else:
                    accuracy = 0
            except:
                accuracy = 0
            
            logger.info(f"✅ LR={lr:.2e} completed: accuracy={accuracy:.4f}, duration={duration:.1f}s")
            
        except Exception as e:
            logger.error(f"❌ LR={lr:.2e} failed: {str(e)}")
            result = {
                'test_id': i + 1,
                'learning_rate': lr,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
    
    logger.info("\n📊 Quick learning rate test results:")
    logger.info("=" * 60)
    
    successful_results = [r for r in results if 'metrics' in r]
    if successful_results:
        # Safe sort
        def get_accuracy(result):
            metrics = result['metrics']
            if 'overall' in metrics and 'accuracy' in metrics['overall']:
                return metrics['overall']['accuracy']
            elif 'overall_accuracy' in metrics:
                return metrics['overall_accuracy']
            return 0
        
        successful_results.sort(key=get_accuracy, reverse=True)
        
        logger.info("🏆 Learning rate ranking:")
        for i, result in enumerate(successful_results[:5]):
            lr = result['learning_rate']
            accuracy = get_accuracy(result)
            duration = result['duration_seconds']
            logger.info(f"  {i+1}. LR={lr:.2e}, accuracy={accuracy:.4f}, time={duration:.1f}s")
        
        best_result = successful_results[0]
        best_lr = best_result['learning_rate']
        best_accuracy = get_accuracy(best_result)
        
        best_config = {
            'best_learning_rate': best_lr,
            'best_accuracy': best_accuracy,
            'all_results': results
        }
        with open('./output/quick_lr_best.json', 'w', encoding='utf-8') as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n🥇 Recommended learning rate: {best_lr:.2e} (accuracy: {best_accuracy:.4f})")
        logger.info(f"💾 Results saved to: ./output/quick_lr_best.json")
    else:
        logger.error("❌ No successful training results")
    
    return len(successful_results) > 0

def focal_lr_search():
    """Focal Loss learning rate search"""
    os.makedirs('./output', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./output/focal_lr_search.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    setup_directories()
    
    logger.info("🎯 Start FireLLM-Classifier Focal Loss learning rate search (Llama 3.1)")
    
    # Ensure model is available
    if not ensure_model_available(DEFAULT_MODEL_PATH, repo_id=DEFAULT_MODEL_REPO, logger=logger):
        logger.error(f"❌ Model not available: {DEFAULT_MODEL_PATH}")
        return False
    
    lr_candidates = [8e-5, 9e-5, 1.0e-4, 1.1e-4, 1.2e-4]
    
    base_config = {
        'model_name': DEFAULT_MODEL_PATH,
        'data_path': DEFAULT_DATA_PATH,
        'filter_baltimore': False,
        'num_epochs': 5,
        'batch_size': 16,
        'use_4bit': True,
        'use_lora': True,
        'lora_r': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.1,
        'gradient_accumulation_steps': 8,
        'weight_decay': 0.005,
        'warmup_ratio': 0.15,
        'min_lr': 5e-7,
        'label_smoothing': 0.05,
        'early_stopping_patience': 5,
        'lr_scheduler': 'cosine_with_restarts',
        'class_weights': False,
        'dropout': 0.15,
        'focal_loss': True,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'max_grad_norm': 1.0,
        'use_cls_head': True,
        'enable_data_balancing': False
    }
    
    results = []
    
    if not os.path.exists(base_config['data_path']):
        logger.error(f"Data file not found: {base_config['data_path']}")
        return False
    
    logger.info(f"📊 Total {len(lr_candidates)} learning rates to test (Focal Loss enabled)")
    logger.info(f"🔧 Focal Loss params: alpha={base_config['focal_alpha']}, gamma={base_config['focal_gamma']}")
    
    for i, lr in enumerate(lr_candidates):
        logger.info(f"🔬 Testing {i+1}/{len(lr_candidates)}: LR={lr:.2e} (Focal Loss enabled)")
        
        try:
            trainer = FireLLMTrainer(
                model_name=base_config['model_name'],
                use_wandb=False,
                project_name="firellm-focal-lr"
            )
            
            train_data, val_data, test_data = trainer.load_and_prepare_data(
                data_path=base_config['data_path'],
                filter_baltimore=base_config['filter_baltimore'],
                save_processed=False
            )
            
            trainer.initialize_model(
                use_4bit=base_config['use_4bit'],
                use_lora=base_config['use_lora'],
                lora_r=base_config['lora_r'],
                lora_alpha=base_config['lora_alpha'],
                lora_dropout=base_config['lora_dropout']
            )
            
            start_time = datetime.now()
            
            trainer.train_model(
                train_data=train_data,
                val_data=val_data,
                save_path=f"./models/focal_lr_llama31_{i+1}",
                learning_rate=lr,
                num_epochs=base_config['num_epochs'],
                batch_size=base_config['batch_size'],
                gradient_accumulation_steps=base_config['gradient_accumulation_steps'],
                weight_decay=base_config['weight_decay'],
                warmup_ratio=base_config['warmup_ratio'],
                min_lr=base_config['min_lr'],
                label_smoothing=base_config['label_smoothing'],
                early_stopping_patience=base_config['early_stopping_patience'],
                lr_scheduler=base_config['lr_scheduler'],
                class_weights=base_config['class_weights'],
                dropout=base_config['dropout'],
                focal_loss=base_config['focal_loss'],
                focal_alpha=base_config['focal_alpha'],
                focal_gamma=base_config['focal_gamma'],
                max_grad_norm=base_config['max_grad_norm'],
                use_cls_head=base_config['use_cls_head']
            )
            
            metrics = trainer.evaluate_model(test_data)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'test_id': i + 1,
                'learning_rate': lr,
                'focal_loss': True,
                'focal_alpha': base_config['focal_alpha'],
                'focal_gamma': base_config['focal_gamma'],
                'metrics': metrics,
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            # Safe access
            try:
                if 'overall' in metrics and 'accuracy' in metrics['overall']:
                    accuracy = metrics['overall']['accuracy']
                elif 'overall_accuracy' in metrics:
                    accuracy = metrics['overall_accuracy']
                else:
                    accuracy = 0
            except:
                accuracy = 0
            
            logger.info(f"✅ LR={lr:.2e} completed: accuracy={accuracy:.4f}, time={duration:.1f}s")
            
            with open('./output/focal_lr_search_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"❌ LR={lr:.2e} failed: {str(e)}")
            result = {
                'test_id': i + 1,
                'learning_rate': lr,
                'focal_loss': True,
                'focal_alpha': base_config['focal_alpha'],
                'focal_gamma': base_config['focal_gamma'],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
    
    logger.info("\n📊 Focal Loss learning rate search results:")
    logger.info("=" * 80)
    
    successful_results = [r for r in results if 'metrics' in r]
    if successful_results:
        def get_accuracy(result):
            metrics = result['metrics']
            if 'overall' in metrics and 'accuracy' in metrics['overall']:
                return metrics['overall']['accuracy']
            elif 'overall_accuracy' in metrics:
                return metrics['overall_accuracy']
            return 0
        
        successful_results.sort(key=get_accuracy, reverse=True)
        
        logger.info("🏆 Focal Loss learning rate ranking:")
        for i, result in enumerate(successful_results[:8]):
            lr = result['learning_rate']
            accuracy = get_accuracy(result)
            duration = result['duration_seconds']
            logger.info(f"  {i+1}. LR={lr:.2e}, accuracy={accuracy:.4f}, time={duration:.1f}s")
        
        best_result = successful_results[0]
        best_lr = best_result['learning_rate']
        best_accuracy = get_accuracy(best_result)
        
        best_config = {
            'best_learning_rate': best_lr,
            'best_accuracy': best_accuracy,
            'focal_loss': True,
            'focal_alpha': base_config['focal_alpha'],
            'focal_gamma': base_config['focal_gamma'],
            'all_results': results
        }
        with open('./output/best_focal_lr_config.json', 'w', encoding='utf-8') as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n🥇 Focal Loss recommended learning rate: {best_lr:.2e} (accuracy: {best_accuracy:.4f})")
        logger.info(f"💾 Results saved to: ./output/best_focal_lr_config.json")
        
        logger.info("\n📈 Comparison suggestions:")
        logger.info("   - If Focal Loss performs better, use it in full training")
        logger.info("   - If similar performance, prefer simpler cross-entropy loss")
        logger.info("   - Note Focal Loss's effect on class imbalance handling")
    else:
        logger.error("❌ No successful training results")
    
    logger.info(f"\n📁 Complete results saved to: ./output/focal_lr_search_results.json")
    return len(successful_results) > 0

def grid_search():
    """Grid search for best parameters"""
    os.makedirs('./output', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./output/grid_search.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    setup_directories()
    
    logger.info("🔍 Start grid search (Llama 3.1)")
    
    # Ensure model is available
    if not ensure_model_available(DEFAULT_MODEL_PATH, repo_id=DEFAULT_MODEL_REPO, logger=logger):
        logger.error(f"❌ Model not available: {DEFAULT_MODEL_PATH}")
        return False
    
    search_space = {
        'learning_rate': [3e-5, 5e-5, 8e-5],
        'batch_size': [8, 16],
        'weight_decay': [0.003, 0.005, 0.008],
        'lr_scheduler': ['cosine_with_restarts', 'cosine'],
        'gradient_accumulation_steps': [4, 8, 12],
        'focal_alpha': [0.2, 0.25, 0.3],
        'focal_gamma': [1.5, 2.0, 2.5]
    }
    
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))
    
    logger.info(f"📊 Total {len(combinations)} combinations to test")
    
    results = []
    
    data_path = DEFAULT_DATA_PATH
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return False
    
    for i, combination in enumerate(combinations):
        config = dict(zip(keys, combination))
        
        logger.info(f"🧪 Testing combination {i+1}/{len(combinations)}: {config}")
        
        try:
            trainer = FireLLMTrainer(
                model_name=DEFAULT_MODEL_PATH,
                use_wandb=False,
                project_name="firellm-gridsearch"
            )
            
            train_data, val_data, test_data = trainer.load_and_prepare_data(
                data_path=data_path,
                filter_baltimore=False,
                save_processed=False
            )
            
            train_data = train_data[:800]
            val_data = val_data[:150]
            
            trainer.initialize_model(
                use_4bit=True,
                use_lora=True,
                lora_r=16,
                lora_alpha=32,
                lora_dropout=0.1
            )
            
            training_config = {
                'num_epochs': 5,
                'warmup_ratio': 0.15,
                'min_lr': 5e-7,
                'label_smoothing': 0.0,
                'early_stopping_patience': 5,
                'class_weights': False,
                'dropout': 0.15,
                'focal_loss': False,
                'max_grad_norm': 1.0,
                'use_cls_head': True
            }
            training_config.update(config)
            
            start_time = datetime.now()
            
            trainer.train_model(
                train_data=train_data,
                val_data=val_data,
                save_path=f"./models/gridsearch_llama31_{i+1}",
                **training_config
            )
            
            metrics = trainer.evaluate_model(test_data[:50])
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'combination_id': i + 1,
                'config': config,
                'metrics': metrics,
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            # Safe access
            try:
                if 'overall' in metrics and 'accuracy' in metrics['overall']:
                    accuracy = metrics['overall']['accuracy']
                elif 'overall_accuracy' in metrics:
                    accuracy = metrics['overall_accuracy']
                else:
                    accuracy = 0
            except:
                accuracy = 0
            
            logger.info(f"✅ Combination {i+1} completed: accuracy={accuracy:.4f}, time={duration:.1f}s")
            
            with open('./output/grid_search_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"❌ Combination {i+1} failed: {str(e)}")
            result = {
                'combination_id': i + 1,
                'config': config,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
    
    logger.info("\n📊 Grid search results:")
    logger.info("=" * 80)
    
    successful_results = [r for r in results if 'metrics' in r]
    if successful_results:
        def get_accuracy(result):
            metrics = result['metrics']
            if 'overall' in metrics and 'accuracy' in metrics['overall']:
                return metrics['overall']['accuracy']
            elif 'overall_accuracy' in metrics:
                return metrics['overall_accuracy']
            return 0
        
        successful_results.sort(key=get_accuracy, reverse=True)
        
        logger.info("🏆 Top 5 best combinations:")
        for i, result in enumerate(successful_results[:5]):
            config = result['config']
            accuracy = get_accuracy(result)
            duration = result['duration_seconds']
            logger.info(f"  {i+1}. Accuracy: {accuracy:.4f}, Time: {duration:.1f}s")
            logger.info(f"     Config: {config}")
            logger.info("")
        
        best_config = successful_results[0]['config']
        with open('./output/best_config.json', 'w', encoding='utf-8') as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Best config saved to: ./output/best_config.json")
        logger.info(f"🎯 Best config: {best_config}")
    else:
        logger.error("❌ No successful training results")
    
    logger.info(f"\n📁 Complete results saved to: ./output/grid_search_results.json")
    return len(successful_results) > 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='FireLLM Training Script (Llama 3.1)')
    parser.add_argument('--mode', choices=['single', 'multi', 'grid', 'lr', 'quick', 'focal', 'test'], default='single',
                       help='Training mode')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Model path for test mode')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Data path for test mode')
    parser.add_argument('--hf_token', '--huggingface_token', type=str, default="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",
                       help='Hugging Face access token')
    
    args = parser.parse_args()
    
    if args.hf_token:
        os.environ['HUGGINGFACE_HUB_TOKEN'] = args.hf_token
    
    print("🔥 FireLLM-Classifier Training Script (Llama 3.1-8B-Instruct)")
    print(f"📋 Mode: {args.mode}")
    
    logger = logging.getLogger(__name__)
    setup_directories()
    
    if not ensure_model_available(DEFAULT_MODEL_PATH, repo_id=DEFAULT_MODEL_REPO, logger=logger):
        logger.warning(f"Base model {DEFAULT_MODEL_PATH} is not available. Some modes may fail.")
    
    if args.mode == 'single':
        single_training()
    elif args.mode == 'multi':
        multi_training()
    elif args.mode == 'grid':
        grid_search()
    elif args.mode == 'lr':
        lr_search()
    elif args.mode == 'quick':
        quick_lr_test()
    elif args.mode == 'focal':
        focal_lr_search()
    elif args.mode == 'test':
        test_mode(model_path=args.model_path, data_path=args.data_path)
    else:
        print("❌ Invalid mode")

if __name__ == "__main__":
    main()