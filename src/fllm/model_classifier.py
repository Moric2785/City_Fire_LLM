import copy
import logging
import os
import json
import torch
import torch.nn as nn
import transformers
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
# 尝试相对导入，失败则使用绝对导入
try:
    from .prompt_generator import FirePromptGenerator, FireIncidentData
except ImportError:
    from prompt_generator import FirePromptGenerator, FireIncidentData

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

IGNORE_INDEX = -100


def set_seed(seed: int = 42):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)





class FireLLMModel:
    
    
    def __init__(self, 
                 model_name: str = "./models/llama3-8b-instruct",
                 use_4bit: bool = True,
                 use_lora: bool = True,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 use_cls_head: bool = True):
        
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_cls_head = use_cls_head
        
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = None
        
        
        self.prompt_generator = FirePromptGenerator()
        
        
        self.fire_spread_classes = {
            2: "fire within the room",
            3: "within the floor", 
            4: "beyond the floor"
        }
        
    def load_model_and_tokenizer(self):
        
        print(f"Loading model: {self.model_name}")
        
        # Configure quantization parameters
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
            
        # Load model
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            max_memory={0: "8GB", "cpu": "16GB"},
            low_cpu_mem_usage=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Debug: Check tokenizer special token settings
        print(f"🔍 Tokenizer debug info:")
        print(f"   EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        print(f"   PAD token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        print(f"   BOS token: {self.tokenizer.bos_token} (ID: {self.tokenizer.bos_token_id})")
        
        # Check for eot_id token
        if hasattr(self.tokenizer, 'eot_id'):
            print(f"   EOT ID: {self.tokenizer.eot_id}")
        
        # Check if vocab contains eot_id
        if '<|eot_id|>' in self.tokenizer.get_vocab():
            print(f"   ✅ Found '<|eot_id|>' token, ID: {self.tokenizer.convert_tokens_to_ids('<|eot_id|>')}")
        
        # Set special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        
        print(f"   🔧 Maintaining Llama3's original token sequence structure, including eot_id")

        
        self.class_special_tokens = ["<S2>", "<S3>", "<S4>"]
        added_num = self.tokenizer.add_special_tokens({
            'additional_special_tokens': [t for t in self.class_special_tokens if t not in self.tokenizer.get_vocab()]
        })
        if added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"✅ Resized embeddings for added special tokens: {self.class_special_tokens}")
        self.class_token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in self.class_special_tokens]
        if any(tid is None or tid < 0 for tid in self.class_token_ids):
            raise RuntimeError(f"Some special tokens not found in vocab: {self.class_special_tokens}")
        print(f"   Class token ids: {self.class_token_ids}")
        
        
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
            
        # Apply LoRA
        if self.use_lora:
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "lm_head"],
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)

            # Print trainable parameter statistics
            trainable_params = 0
            all_params = 0
            for _, param in self.model.named_parameters():
                numel = param.numel()
                all_params += numel
                if param.requires_grad:
                    trainable_params += numel
            print(
                f"Trainable params: {trainable_params} || All params: {all_params} || "
                f"Trainable%: {100 * trainable_params / max(1, all_params):.2f}"
            )
            
        print(f"Model loaded successfully on {self.device}")

        # Unfreeze output head for new tokens to learn quickly (LoRA freezes base parameters by default)
        try:
            output_emb = self.model.get_output_embeddings()
            if output_emb is not None:
                for param in output_emb.parameters():
                    param.requires_grad = True
                print("✅ Unfroze lm_head for training")
        except Exception as e:
            print(f"⚠️ Failed to unfreeze lm_head: {e}")

        # Initialize explicit classification head (optional, enabled by default)
        try:
            if self.use_cls_head:
                hidden_size = getattr(self.model.config, 'hidden_size', None)
                if hidden_size is None and hasattr(self.model.config, 'hidden_sizes'):
                    hidden_size = self.model.config.hidden_sizes[-1]
                if hidden_size is None:
                    raise ValueError("Cannot get hidden_size for classification head")
                
                # Enhanced classification head: add dropout and LayerNorm
                self.classifier = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size // 2, 3)
                ).to(self.device)
                print(f"✅ Initialized enhanced classification head: hidden_size={hidden_size} -> {hidden_size//2} -> 3")
        except Exception as e:
            print(f"⚠️ Classification head initialization failed: {e}")
        
    def create_fire_prompt(self, incident_data: FireIncidentData) -> str:
        """Create fire incident prompt using unified prompt generator"""
        return self.prompt_generator.create_fire_prompt(incident_data)
        
    def create_training_data(self, incident_data: FireIncidentData) -> Dict[str, torch.Tensor]:
        """Create training data - candidate logits 3-class classification (without target token in input)"""
        # Build prompt up to "Answer: " (without target token)
        prompt = self.create_fire_prompt(incident_data)
        prompt = prompt.split("Answer:")[0] + "Answer: "

        # Class index (2,3,4 -> 0,1,2)
        class_label = int(incident_data.fire_spread_label) - 2
        if class_label < 0 or class_label > 2:
            print(f"⚠️ Invalid class: {incident_data.fire_spread_label}")
            return None

        # Encode prompt (without target token)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
            return_attention_mask=True
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "class_label": torch.tensor(class_label, dtype=torch.long)
        }
        
    def train(self, 
              train_data: List[FireIncidentData],
              val_data: List[FireIncidentData],
              num_epochs: Optional[int] = None,
              batch_size: Optional[int] = None,
              learning_rate: Optional[float] = None,
              save_path: Optional[str] = None,
              weight_decay: Optional[float] = None,
              gradient_accumulation_steps: Optional[int] = None,
              lr_scheduler: Optional[str] = None,
              warmup_ratio: Optional[float] = None,
              min_lr: Optional[float] = None,
              early_stopping_patience: Optional[int] = None,
              dropout: Optional[float] = None,
              **kwargs):
        """训练模型 - 使用简单的训练循环避免兼容性问题。
        超参数默认从 self.training_config 读取（由外部 trainer/train.py 设置）。"""
        
        if self.model is None:
            self.load_model_and_tokenizer()
        
        # 来自外部配置的默认值
        cfg = getattr(self, 'training_config', {}) if hasattr(self, 'training_config') else {}
        num_epochs = num_epochs if num_epochs is not None else cfg.get('num_epochs', 5)
        batch_size = batch_size if batch_size is not None else cfg.get('batch_size', 16)
        learning_rate = learning_rate if learning_rate is not None else cfg.get('learning_rate', 3e-4)
        save_path = save_path if save_path is not None else cfg.get('save_path', "./models/firellm_finetuned")
        weight_decay = weight_decay if weight_decay is not None else cfg.get('weight_decay', 0.0001)
        gradient_accumulation_steps = gradient_accumulation_steps if gradient_accumulation_steps is not None else cfg.get('gradient_accumulation_steps', 2)
        early_stopping_patience = early_stopping_patience if early_stopping_patience is not None else cfg.get('early_stopping_patience', 3)
        print(f"🔧 有效训练超参数: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}, weight_decay={weight_decay}")
            
        # 准备训练数据
        train_dataset = [self.create_training_data(data) for data in train_data]
        val_dataset = [self.create_training_data(data) for data in val_data]
        
        # 过滤掉None的样本
        train_dataset = [sample for sample in train_dataset if sample is not None]
        val_dataset = [sample for sample in val_dataset if sample is not None]
        
        if len(train_dataset) == 0:
            raise ValueError("没有有效的训练样本！请检查数据准备逻辑。")
        
        print(f"✅ 有效训练样本: {len(train_dataset)}/{len(train_data)}")
        print(f"✅ 有效验证样本: {len(val_dataset)}/{len(val_data)}")
        
        # 简要检查数据长度分布
        print("🔍 训练/验证序列长度概览...")
        train_lengths = [len(sample['input_ids']) for sample in train_dataset]
        val_lengths = [len(sample['input_ids']) for sample in val_dataset]
        print(f"   训练长度: min={min(train_lengths)}, max={max(train_lengths)}, avg={sum(train_lengths)/len(train_lengths):.1f}")
        print(f"   验证长度: min={min(val_lengths)}, max={max(val_lengths)}, avg={sum(val_lengths)/len(val_lengths):.1f}")
        
        # 计算类别权重（按训练集频次的逆频权重）
        # 检查是否启用类别权重（从training_config获取）
        use_class_weights = getattr(self, 'training_config', {}).get('class_weights', False)
        
        if use_class_weights:
            try:
                num_classes = 3
                class_counts = [0 for _ in range(num_classes)]
                for sample in train_dataset:
                    lab = int(sample['class_label']) if torch.is_tensor(sample['class_label']) else int(sample['class_label'])
                    if 0 <= lab < num_classes:
                        class_counts[lab] += 1
                # 防止除零
                class_counts = [c if c > 0 else 1 for c in class_counts]
                total_samples = sum(class_counts)
                class_weights_list = [total_samples / (num_classes * c) for c in class_counts]
                class_weights = torch.tensor(class_weights_list, dtype=torch.float, device=self.device)
                print(f"🔧 类别频次: {class_counts} -> 类别权重: {[round(w, 4) for w in class_weights_list]}")
            except Exception as e:
                print(f"⚠️ 计算类别权重失败: {e}，将使用等权重")
                class_weights = None
        else:
            print(f"🔧 数据已平衡，不使用类别权重")
            class_weights = None
        
        # 设置优化器 - 同时包含分类头参数（若启用）
        optim_params = [
            {"params": [p for p in self.model.parameters() if p.requires_grad]},
        ]
        if self.use_cls_head and self.classifier is not None:
            optim_params.append({"params": [p for p in self.classifier.parameters() if p.requires_grad]})
        optimizer = torch.optim.AdamW(
            optim_params, 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 使用更合适的学习率调度器
        from torch.optim.lr_scheduler import OneCycleLR
        
        # 配置梯度累积（声明在此以便计算调度步数）
        accumulation_steps = gradient_accumulation_steps  # 使用传入的梯度累积步数
        
        # 计算优化器实际更新的总步数（考虑batch与梯度累积）
        total_batches = (len(train_dataset) + batch_size - 1) // batch_size
        updates_per_epoch = (total_batches + accumulation_steps - 1) // accumulation_steps
        total_steps = max(1, updates_per_epoch * num_epochs)
        
        # OneCycleLR配置（方案一）
        div_factor = 10.0       # 初始学习率 = max_lr / 10
        final_div_factor = 1e2  # 最终学习率 = max_lr / 100
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.1,  # 10%的时间用于warmup
            anneal_strategy='cos',
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
        
        print(f"✅ 使用OneCycleLR调度器:")
        print(f"   总训练步数(优化器更新数): {total_steps}")
        print(f"   最大学习率: {learning_rate}")
        print(f"   初始学习率: {learning_rate / div_factor:.2e}")
        print(f"   最终学习率: {learning_rate / final_div_factor:.2e}")
        
        # 训练循环
        print("Starting training...")
        self.model.train()
        
        # 初始化wandb（如果可用）
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            import wandb
            if wandb.run is None:
                wandb.init(
                    project=f"firellm_{timestamp}",
                    config={
                        "model_name": self.model_name,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "epochs": num_epochs,
                        "use_lora": self.use_lora,
                        "lora_r": self.lora_r,
                        "lora_alpha": self.lora_alpha,
                        "lora_dropout": self.lora_dropout
                    }
                )
                print("✅ Wandb initialized for training tracking")
            use_wandb = True
        except ImportError:
            use_wandb = False
            print("⚠️ Wandb not available, skipping logging")
        
        # 添加梯度累积
        optimizer.zero_grad()
        
        # 早停相关变量
        best_val_loss = float('inf')
        patience_counter = 0
        
        import random
        for epoch in range(num_epochs):
            # 在每个epoch开始随机打乱训练样本，避免难样本成簇造成loss尖峰
            random.shuffle(train_dataset)
            total_loss = 0
            num_batches = 0
            
            # 计算总批次数
            total_batches = (len(train_dataset) + batch_size - 1) // batch_size
            
            # 使用tqdm显示训练进度
            pbar = tqdm(range(0, len(train_dataset), batch_size), 
                       desc=f"Epoch {epoch+1}/{num_epochs}", 
                       total=total_batches,
                       unit="batch")
            
            for batch_idx in pbar:
                batch = train_dataset[batch_idx:batch_idx+batch_size]
                
                # 准备批次数据
                input_ids_list = []
                attention_mask_list = []

                for item in batch:
                    ids = item['input_ids'].tolist() if torch.is_tensor(item['input_ids']) else item['input_ids']
                    mask = item['attention_mask'].tolist() if torch.is_tensor(item['attention_mask']) else item['attention_mask']
                    input_ids_list.append(ids)
                    attention_mask_list.append(mask)

                # 填充到相同长度
                max_len = max(len(ids) for ids in input_ids_list)
                padded_input_ids = [ids + [self.tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids_list]
                padded_attention_mask = [mask + [0] * (max_len - len(mask)) for mask in attention_mask_list]

                # 转换为张量
                input_ids = torch.tensor(padded_input_ids, dtype=torch.long, device=self.device)
                attention_mask = torch.tensor(padded_attention_mask, dtype=torch.long, device=self.device)

                # 前向传播：输出隐藏态或logits
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)

                # 取每个样本最后一个非PAD位置
                last_indices = attention_mask.sum(dim=1) - 1  # [B]
                batch_indices = torch.arange(input_ids.size(0), device=self.device)  # [B]

                if self.use_cls_head and self.classifier is not None:
                    # 用显式分类头
                    last_hidden = outputs.hidden_states[-1]  # [B, L, H]
                    last_hidden_vec = last_hidden[batch_indices, last_indices, :]  # [B, H]
                    target_logits = self.classifier(last_hidden_vec)  # [B, 4]
                else:
                    # 回退：token法
                    logits = outputs.logits  # [B, L, V]
                    last_logits = logits[batch_indices, last_indices, :]  # [B, V]
                    if isinstance(self.class_token_ids, list):
                        class_token_ids = torch.tensor(self.class_token_ids, dtype=torch.long, device=self.device)
                    else:
                        class_token_ids = self.class_token_ids.to(self.device)
                    target_logits = last_logits.index_select(dim=1, index=class_token_ids)  # [B, 4]
                
                # 将原始样本的类别索引收集为Tensor（已在create_training_data中映射为0..2）
                class_indices = []
                for item in batch:
                    lab = int(item['class_label']) if torch.is_tensor(item['class_label']) else int(item['class_label'])
                    if not (0 <= lab <= 2):
                        raise ValueError(f"Unknown class label (expect 0..2): {lab}")
                    class_indices.append(lab)
                class_labels = torch.tensor(class_indices, dtype=torch.long, device=self.device)
                
                # 调试信息（可启用）：类别示例（0..2 -> 2..4）
                # if len(class_indices) > 0:
                #     logging.debug(f"class samples (idx->label): {class_indices[:3]} -> {[x+2 for x in class_indices[:3]]}")
                
                # 3分类交叉熵损失（根据配置决定是否使用权重）
                if class_weights is not None and use_class_weights:
                    try:
                        cw = class_weights.detach().cpu().tolist()
                        # 开方平滑、均值归一、截断
                        import math
                        cw = [math.sqrt(max(1e-12, w)) for w in cw]
                        mean_w = sum(cw) / max(1, len(cw))
                        cw = [w / max(1e-12, mean_w) for w in cw]
                        cw = [min(1.8, max(0.6, w)) for w in cw]
                        class_weights_smoothed = torch.tensor(cw, dtype=torch.float, device=self.device)
                        print(f"   🔧 Smoothed class weights: {[round(x,3) for x in cw]}")
                        loss_fct = nn.CrossEntropyLoss(weight=class_weights_smoothed, label_smoothing=0.05)
                    except Exception:
                        loss_fct = nn.CrossEntropyLoss(label_smoothing=0.05)
                else:
                    loss_fct = nn.CrossEntropyLoss(label_smoothing=0.05)
                loss = loss_fct(target_logits, class_labels)
                
                # 详细调试信息（静默模式，只保留关键信息）
                # 删除大量调试输出，只保留核心训练逻辑
                
                # 检查loss是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ 检测到无效loss: {loss.item()}, 跳过此batch")
                    continue
                
                # 检查loss值是否合理
                if loss.item() > 20.0:
                    print(f"⚠️ Loss值过高: {loss.item():.4f}, 可能需要调整学习率")
                
                # 反向传播 - 使用梯度累积
                (loss / accumulation_steps).backward()
                
                # 每accumulation_steps步更新一次参数
                if (batch_idx // batch_size + 1) % accumulation_steps == 0:
                    # 检查梯度是否有效
                    params_to_clip = list(self.model.parameters()) + (list(self.classifier.parameters()) if (self.use_cls_head and self.classifier is not None) else [])
                    grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"⚠️ 检测到无效梯度范数: {grad_norm.item()}")
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 更新学习率（OneCycleLR需要每步更新）
                    scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 更新进度条显示当前损失
                current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else learning_rate
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg_Loss': f'{total_loss/num_batches:.4f}',
                    'LR': f'{current_lr:.6f}'
                })
                
                # 记录到wandb（每10个batch记录一次，避免日志过多）
                if use_wandb and (batch_idx // batch_size) % 10 == 0:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch": batch_idx // batch_size + epoch * total_batches,
                        "learning_rate": current_lr
                    })
            
            # 确保最后一个batch的梯度也被更新
            if num_batches % accumulation_steps != 0:
                params_to_clip = list(self.model.parameters()) + (list(self.classifier.parameters()) if (self.use_cls_head and self.classifier is not None) else [])
                grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
                if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    optimizer.step()
                    scheduler.step()  # 更新学习率
                optimizer.zero_grad()
            
            # 不再需要手动更新学习率，因为OneCycleLR已经每步更新了
            # scheduler.step()  # 删除这行
            
            # 计算平均损失
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # 记录epoch级别的指标到wandb
            if use_wandb:
                current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else learning_rate
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "learning_rate": current_lr
                })
            
            # 验证
            if val_dataset:
                self.model.eval()
                val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for i in range(0, len(val_dataset), batch_size):
                        batch = val_dataset[i:i+batch_size]
                        
                        # 准备批次数据（与训练相同）
                        input_ids_list = []
                        attention_mask_list = []

                        for item in batch:
                            ids = item['input_ids'].tolist() if torch.is_tensor(item['input_ids']) else item['input_ids']
                            mask = item['attention_mask'].tolist() if torch.is_tensor(item['attention_mask']) else item['attention_mask']
                            input_ids_list.append(ids)
                            attention_mask_list.append(mask)

                        max_len = max(len(ids) for ids in input_ids_list)
                        padded_input_ids = [ids + [self.tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids_list]
                        padded_attention_mask = [mask + [0] * (max_len - len(mask)) for mask in attention_mask_list]

                        input_ids = torch.tensor(padded_input_ids, dtype=torch.long, device=self.device)
                        attention_mask = torch.tensor(padded_attention_mask, dtype=torch.long, device=self.device)

                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
                        last_indices = attention_mask.sum(dim=1) - 1  # [B]
                        batch_indices = torch.arange(input_ids.size(0), device=self.device)  # [B]

                        if self.use_cls_head and self.classifier is not None:
                            last_hidden = outputs.hidden_states[-1]  # [B, L, H]
                            last_hidden_vec = last_hidden[batch_indices, last_indices, :]  # [B, H]
                            target_logits = self.classifier(last_hidden_vec)  # [B, 4]
                        else:
                            logits = outputs.logits  # [B, L, V]
                            last_logits = logits[batch_indices, last_indices, :]  # [B, V]
                            if isinstance(self.class_token_ids, list):
                                class_token_ids = torch.tensor(self.class_token_ids, dtype=torch.long, device=self.device)
                            else:
                                class_token_ids = self.class_token_ids.to(self.device)
                            target_logits = last_logits.index_select(dim=1, index=class_token_ids)  # [B, 4]
                        
                        # 将原始样本的类别索引收集为Tensor（已在create_training_data中映射为0..2）
                        class_indices = []
                        for item in batch:
                            lab = int(item['class_label']) if torch.is_tensor(item['class_label']) else int(item['class_label'])
                            if not (0 <= lab <= 2):
                                raise ValueError(f"Unknown class label (expect 0..2): {lab}")
                            class_indices.append(lab)
                        class_labels = torch.tensor(class_indices, dtype=torch.long, device=self.device)
                        
                        # 3分类交叉熵损失（验证阶段与训练阶段保持一致）
                        if class_weights is not None and use_class_weights:
                            try:
                                cw = class_weights.detach().cpu().tolist()
                                import math
                                cw = [math.sqrt(max(1e-12, w)) for w in cw]
                                mean_w = sum(cw) / max(1, len(cw))
                                cw = [w / max(1e-12, mean_w) for w in cw]
                                cw = [min(1.8, max(0.6, w)) for w in cw]
                                class_weights_smoothed = torch.tensor(cw, dtype=torch.float, device=self.device)
                                loss_fct = nn.CrossEntropyLoss(weight=class_weights_smoothed, label_smoothing=0.05)
                            except Exception:
                                loss_fct = nn.CrossEntropyLoss(label_smoothing=0.05)
                        else:
                            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.05)
                        batch_loss = loss_fct(target_logits, class_labels)

                        val_loss += batch_loss.item()
                        val_batches += 1
                
                if val_batches > 0:
                    avg_val_loss = val_loss / val_batches
                    print(f"Validation loss: {avg_val_loss:.4f}")
                    
                    # 早停检查
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        print(f"✅ 验证损失改善，保存最佳模型 (val_loss: {avg_val_loss:.4f})")
                    else:
                        patience_counter += 1
                        print(f"⚠️ 验证损失未改善 ({patience_counter}/{early_stopping_patience})")
                        
                        # 检查是否触发早停
                        if patience_counter >= early_stopping_patience:
                            print(f"🛑 早停触发！在epoch {epoch+1}停止训练")
                            break
                    
                    # 记录验证损失到wandb
                    if use_wandb:
                        wandb.log({
                            "epoch": epoch + 1,
                            "val_loss": avg_val_loss,
                            "best_val_loss": best_val_loss,
                            "patience_counter": patience_counter
                        })
                else:
                    print("Warning: No validation batches processed")
                self.model.train()
        
        # 保存模型
        os.makedirs(save_path, exist_ok=True)
        
        # 保存LoRA权重
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(save_path)
            print(f"LoRA weights saved to {save_path}")
        else:
            # 如果没有LoRA，保存完整模型
            torch.save(self.model.state_dict(), f"{save_path}/pytorch_model.bin")
            print(f"Full model weights saved to {save_path}")
        


        # 额外保存分类头（若启用）
        if self.use_cls_head and self.classifier is not None:
            torch.save(self.classifier.state_dict(), os.path.join(save_path, "classifier_head.bin"))
            print(f"Classifier head saved to {os.path.join(save_path, 'classifier_head.bin')}")

        # 保存tokenizer
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
        
        # 完成wandb运行
        if use_wandb:
            try:
                wandb.finish()
                print("✅ Wandb run completed")
            except Exception as e:
                print(f"⚠️ Error finishing wandb run: {e}")
        
    def predict(self, incident_data: FireIncidentData) -> Dict:
        """预测火灾后果 - 候选Logits三分类（3选1）"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
            
        # 创建提示词（不包含答案类别）
        prompt = self.create_fire_prompt(incident_data)
        prompt = prompt.split("Answer:")[0].strip() + "\n\nAnswer: "
        
        # 候选softmax分类
        prediction = self._classify_with_logits(prompt)
        
        return {
            "input_prompt": prompt,
            "generated_text": "",
            "generated_completion": "",
            "prediction": prediction
        }

    def _classify_with_logits(self, prompt: str) -> Dict:
        """候选Logits三分类：取最后位置logits，对3个特殊token做softmax"""
        import torch
        import torch.nn.functional as F

        # 确保推理过程中的随机性被控制
        set_seed(42)
        
        self.model.eval()
        # 编码prompt
        batch = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=2048,
            return_attention_mask=True
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            last_index = int(attention_mask.sum(dim=1).item() - 1)
            if self.use_cls_head and self.classifier is not None:
                last_hidden = outputs.hidden_states[-1]  # [1, L, H]
                last_hidden_vec = last_hidden[:, last_index, :]  # [1, H]
                target_logits = self.classifier(last_hidden_vec)  # [1, 4]
            else:
                logits = outputs.logits  # [1, L, V]
                last_logits = logits[:, last_index, :]  # [1, V]
                if isinstance(self.class_token_ids, list):
                    class_token_ids = torch.tensor(self.class_token_ids, dtype=torch.long, device=self.device)
                else:
                    class_token_ids = self.class_token_ids.to(self.device)
                target_logits = last_logits.index_select(dim=1, index=class_token_ids)  # [1, 4]

            probs_tensor = F.softmax(target_logits[0], dim=-1)
            probs_list = probs_tensor.tolist()

        class_ids = [2,3,4]
        probs = {class_ids[i]: float(probs_list[i]) for i in range(3)}
        predicted_idx = int(torch.argmax(probs_tensor).item())
        predicted_class = class_ids[predicted_idx]

        return {
            "fire_spread": {
                "predicted_class": predicted_class,
                "class_name": self.fire_spread_classes.get(predicted_class, "unknown"),
                "probabilities": probs
            }
        }
        
    def _parse_prediction(self, generated_text: str) -> Dict:
        """解析生成的预测文本（仅解析模型在“Answer:”之后的新生成内容）"""
        
        text = (generated_text or "").strip()
        print(f"\n🔍 生成的文本(仅completion): {text}")
        
        import re
        # 1) 优先按模板严格解析：
        fire_spread_match = None
        casualty_match = None
        
        m_spread = re.search(r"fire\s*spread\s*status\s*:\s*<\s*([2-4])\s*>", text, re.IGNORECASE)
        if m_spread:
            fire_spread_match = int(m_spread.group(1))
            print(f"✅ 模板解析到火灾扩散类别: {fire_spread_match}")
        
        m_inj = re.search(r"number\s*of\s*injuries\s*:\s*<\s*([0-2])\s*>", text, re.IGNORECASE)
        if m_inj:
            casualty_match = int(m_inj.group(1))
            print(f"✅ 模板解析到伤亡(受伤)类别: {casualty_match}")
        
        # 2) 尝试角括号数字（带上下文关键词约束）：
        if fire_spread_match is None:
            m_ctx_s = re.search(r"(spread|floor|building)[^<]*<\s*([2-4])\s*>", text, re.IGNORECASE)
            if m_ctx_s:
                fire_spread_match = int(m_ctx_s.group(2))
                print(f"✅ 上下文角括号解析到扩散类别: {fire_spread_match}")
        
        # 3) 备用：严格的独立数字（词边界）
        if fire_spread_match is None:
            nums = re.findall(r"\b([2-4])\b", text)
            if nums:
                fire_spread_match = int(nums[0])
                print(f"🔍 备用数字解析到扩散类别: {fire_spread_match}")
        
        # 4) 兜底默认值
        if fire_spread_match is None:
            fire_spread_match = 2
            print(f"⚠️ 使用默认火灾扩散类别: {fire_spread_match}")
        
        return {
            "fire_spread": {
                "predicted_class": fire_spread_match,
                "class_name": self.fire_spread_classes.get(fire_spread_match, "unknown"),
                "probabilities": self._estimate_probabilities(fire_spread_match, len(self.fire_spread_classes))
            }
        }
        
    def _estimate_probabilities(self, predicted_class: int, num_classes: int) -> Dict[int, float]:
        """估计类别概率（简化版本）"""
        if predicted_class is None:
            return {i: 1.0/num_classes for i in range(2, 2+num_classes)}
            
        probs = {i: 0.1 for i in range(2, 2+num_classes)}
        probs[predicted_class] = 0.7
        return probs
        
    def _calculate_consequence(self, fire_spread_class: int, casualty_class: int) -> Tuple[int, float, List[str]]:
        """计算火灾后果等级、风险评分和建议措施"""
        
        # 火灾后果等级 (1-5)
        # 1: 轻微, 2: 轻微-中等, 3: 中等, 4: 中等-严重, 5: 严重
        if fire_spread_class == 2 and casualty_class == 0:
            consequence_level = 1  # 房间内，无伤亡
        elif fire_spread_class == 2 and casualty_class in [1, 2]:
            consequence_level = 2  # 房间内，有伤亡
        elif fire_spread_class == 3 and casualty_class == 0:
            consequence_level = 2  # 楼层内，无伤亡
        elif fire_spread_class == 3 and casualty_class in [1, 2]:
            consequence_level = 3  # 楼层内，有伤亡
        elif fire_spread_class == 4 and casualty_class == 0:
            consequence_level = 4  # 楼内/楼外，无伤亡
        elif fire_spread_class == 4 and casualty_class in [1, 2]:
            consequence_level = 5  # 楼内/楼外，有伤亡
        else:
            consequence_level = 3  # 默认中等
        
        # 风险评分 (0.0-1.0)
        fire_spread_weight = 0.6
        casualty_weight = 0.4
        
        # 火灾扩散风险 (2=0.1, 3=0.3, 4=0.8，其中4包含原来的4和5)
        fire_spread_risk = {2: 0.1, 3: 0.3, 4: 0.8}.get(fire_spread_class, 0.5)
        
        # 伤亡风险 (0=0.0, 1=0.5, 2=0.9)
        casualty_risk = {0: 0.0, 1: 0.5, 2: 0.9}.get(casualty_class, 0.5)
        
        risk_score = fire_spread_weight * fire_spread_risk + casualty_weight * casualty_risk
        
        # 建议措施
        recommendations = self._get_recommendations(consequence_level, fire_spread_class, casualty_class)
        
        return consequence_level, risk_score, recommendations
        
    def _get_severity_description(self, consequence_level: int) -> str:
        """获取严重程度描述"""
        descriptions = {
            1: "轻微 - 火灾影响有限，风险较低",
            2: "轻微-中等 - 需要关注，有一定风险",
            3: "中等 - 火灾影响中等，需要及时响应",
            4: "中等-严重 - 火灾影响较大，需要紧急响应",
            5: "严重 - 火灾影响严重，需要立即紧急响应"
        }
        return descriptions.get(consequence_level, "未知")
        
    def _get_recommendations(self, consequence_level: int, fire_spread_class: int, casualty_class: int) -> List[str]:
        """获取建议措施"""
        recommendations = []
        
        # 基于后果等级的建议
        if consequence_level <= 2:
            recommendations.append("加强日常防火检查")
            recommendations.append("确保消防设备正常工作")
        elif consequence_level == 3:
            recommendations.append("立即启动应急响应")
            recommendations.append("疏散相关区域人员")
            recommendations.append("联系消防部门")
        elif consequence_level >= 4:
            recommendations.append("立即启动最高级别应急响应")
            recommendations.append("全面疏散建筑内人员")
            recommendations.append("紧急联系消防部门")
            recommendations.append("准备医疗救援")
        
        # 基于火灾扩散的建议
        if fire_spread_class >= 4:
            recommendations.append("检查防火分区是否有效")
            recommendations.append("评估建筑结构安全性")
        
        # 基于伤亡情况的建议
        if casualty_class >= 1:
            recommendations.append("准备医疗救援")
            recommendations.append("联系急救部门")
        
        return recommendations
        
    def evaluate(self, test_data: List[FireIncidentData]) -> Dict:
        """评估模型性能"""
        
        predictions = []
        true_labels = []
        
        # 添加进度条
        from tqdm import tqdm
        print(f"\n🔍 开始评估 {len(test_data)} 个测试样本...")
        
        for data in tqdm(test_data, desc="评估进度", unit="样本"):
            pred = self.predict(data)
            predictions.append(pred["prediction"])
            true_labels.append({
                "fire_spread": data.fire_spread_label,
            })
            
        # 计算准确率
        fire_spread_acc = self._calculate_accuracy(
            [p["fire_spread"]["predicted_class"] for p in predictions],
            [l["fire_spread"] for l in true_labels]
        )
        
        return {
            "fire_spread_accuracy": fire_spread_acc,
            "overall_accuracy": fire_spread_acc,
            "predictions": predictions,
            "true_labels": true_labels
        }

    def evaluate_rag(self, test_data: List[FireIncidentData], rag_model) -> Dict:
        """使用RAG模型评估性能"""
        
        predictions = []
        true_labels = []
        
        # 添加进度条
        from tqdm import tqdm
        print(f"\n🔍 开始RAG评估 {len(test_data)} 个测试样本...")
        
        for data in tqdm(test_data, desc="RAG评估进度", unit="样本"):
            pred = rag_model.predict(data)
            predictions.append(pred.get("prediction", {"fire_spread": {"predicted_class": -1}}))
            true_labels.append({
                "fire_spread": data.fire_spread_label,
            })
            
        # 计算准确率
        fire_spread_acc = self._calculate_accuracy(
            [p["fire_spread"]["predicted_class"] for p in predictions],
            [l["fire_spread"] for l in true_labels]
        )
        
        return {
            "fire_spread_accuracy": fire_spread_acc,
            "overall_accuracy": fire_spread_acc,
            "predictions": predictions,
            "true_labels": true_labels
        }
        
    def _calculate_accuracy(self, predictions: List, true_labels: List) -> float:
        """计算准确率"""
        if not predictions or not true_labels:
            return 0.0
            
        correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        return correct / len(predictions)

def test_data_preparation():
    """测试数据准备逻辑"""
    print("🧪 测试数据准备逻辑...")
    
    # 创建测试数据
    test_data = FireIncidentData(
        incident_key="test_001",
        building_info={
            'state': 'MD',
            'zip_code': '21201',
            'occupant_type': 'Residential',
            'stories_above': '3',
            'stories_below': '1',
            'square_footage': '5000',
            'build_year': '1990',
            'build_material': 'Wood'
        },
        weather_info={
            'temperature': '25',
            'humidity': '60'
        },
        fire_info={
            'origin': 'Kitchen',
            'time': '14:30',
            'date': '2023-01-15',
            'heat_source': 'Stove',
            'ignited_material': 'Grease',
            'floor': '1',
            'detector_status': 'Working',
            'ase_status': 'Working',
            'response_time': '5'
        },
        community_context={
            'median_income': '50000',
            'median_rent': '1200',
            'housing_occupancy': '85',
            'bachelor_degree': '30',
            'elderly_population': '15',
            'employment_rate': '70',
            'black_population': '25',
            'coal_wood_heating': '5'
        },
        fire_spread_label=3
    )
    
    # 创建模型实例
    model = FireLLMModel()
    
    # 测试提示词创建
    prompt = model.create_fire_prompt(test_data)
    print(f"✅ 提示词创建成功，长度: {len(prompt)}")
    
    # 测试训练数据创建（分类头）
    try:
        # 先加载模型和tokenizer
        model.load_model_and_tokenizer()
        training_data = model.create_training_data(test_data)
        print(f"✅ 训练数据创建成功")
        print(f"   输入长度: {len(training_data['input_ids'])}")
        print(f"   标签长度: {len(training_data['labels'])}")
        print(f"   有效标签数量: {torch.sum(training_data['labels'] != IGNORE_INDEX).item()}")
            
    except Exception as e:
        print(f"❌ 训练数据创建失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("🧪 测试完成")

    # ... 创建 test_data ...

if __name__ == "__main__":
    test_data_preparation()
