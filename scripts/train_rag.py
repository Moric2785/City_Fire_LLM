#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FireLLM-Classifier 训练脚本 - 支持单个配置、网格搜索和RAG训练
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

from fllm.trainer_classifier3 import FireLLMTrainer
from fllm.utils import setup_directories

def run_training(config: dict, mode: str):
    """通用训练函数"""
    # 设置日志
    log_file = f'./output/training_{mode}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    
    setup_directories()
    
    logger.info(f"Start FireLLM-Classifier training in '{mode}' mode")
    logger.debug(f"Config: {json.dumps(config, indent=2)}")

    try:
        if not os.path.exists(config['data_path']):
            logger.error(f"Data file not found: {config['data_path']}")
            return
        
        if 'model_name' in config and not os.path.exists(config['model_name']):
             logger.error(f"Model not found: {config['model_name']}")
             return

        # 初始化训练器
        trainer = FireLLMTrainer(
            model_name=config.get('model_name', "/home/yangzongxian/xlz/models/llama3-8b-instruct"),
            use_wandb=True,
            project_name=f"firellm-classifier-{mode}"
        )
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"./models/firellm_{mode}_{timestamp}"
        
        # 运行完整流程
        # 从config中移除已经明确传递的参数，避免重复
        config_copy = config.copy()
        config_copy.pop('data_path', None)
        config_copy.pop('use_rag', None)
        
        trainer.run_complete_training_pipeline(
            data_path=config['data_path'],
            use_rag=config.get('use_rag', False),
            save_path=model_save_path,
            **config_copy
        )

        logger.info(f"Training finished successfully for mode '{mode}'.")

    except Exception as e:
        logger.error(f"训练失败: {str(e)}", exc_info=True)
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FireLLM 训练脚本')
    parser.add_argument('--mode', choices=['single', 'rag'], default='single',
                       help='训练模式: single=标准训练, rag=RAG增强训练')
    # RAG特定参数
    parser.add_argument('--kg-nodes', type=str, default='./data/kg_nodes.csv', help='知识图谱节点文件路径')
    parser.add_argument('--kg-edges', type=str, default='./data/kg_edges.csv', help='知识图谱边文件路径')
    
    args = parser.parse_args()
    
    print(f"🔥 FireLLM-Classifier 训练脚本 | 模式: {args.mode}")

    # 基础配置
    config = {
        'model_name': "/home/yangzongxian/xlz/models/llama3-8b-instruct",
        'data_path': "./data/merged_19_22_data_4class.csv",
        'filter_baltimore': False,
        'num_epochs': 10,
        'batch_size': 16,
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
        'max_grad_norm': 1.0,
        'use_cls_head': True,
    }

    if args.mode == 'rag':
        config.update({
            'use_rag': True,
            'kg_nodes_path': args.kg_nodes,
            'kg_edges_path': args.kg_edges,
            # RAG特定超参数可以放在这里
            'kg_epochs': 30,
            'retrieval_k': 3
        })

    run_training(config, args.mode)

if __name__ == "__main__":
    main()
