import os
from huggingface_hub import snapshot_download, login

# ===============================
# 1. 在这里放你的 Hugging Face Token
# ===============================
HF_TOKEN = "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"

# （可选）也可以从环境变量读取，更安全
# HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("请先设置 Hugging Face Access Token")

# ===============================
# 2. 登录 Hugging Face（一次即可）
# ===============================
login(token=HF_TOKEN)

# ===============================
# 3. 下载模型到本地
# ===============================
snapshot_download(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    local_dir="./models/llama3.1-8b-instruct",
    local_dir_use_symlinks=False,
    resume_download=True,   # 断点续传
)

print("✅ LLaMA 3.1 8B Instruct 下载完成")

