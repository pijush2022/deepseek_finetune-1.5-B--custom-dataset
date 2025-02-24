# deepseek_finetune-1.5-B--custom-dataset



# DeepSeek 1.5B Fine-Tuning

This repository contains resources and code for fine-tuning the **DeepSeek-R1-Distill-Qwen-1.5B** language model using **Low-Rank Adaptation (LoRA)**. The goal is to adapt the model to specific tasks using the **Ancient-Indian-Wisdom** dataset.

## 📊 Fine-Tuning Technique

### ✅ **Low-Rank Adaptation (LoRA):**
- Efficient fine-tuning method that introduces low-rank matrices into attention layers.
- **LoRA Configuration:**
  - `r = 16` → Rank of low-rank matrices
  - `lora_alpha = 32` → Scaling factor
  - `lora_dropout = 0.1` → Dropout for regularization
  - `bias = "none"` → No bias for LoRA layers
  - **Target Modules:** `["q_proj", "k_proj", "v_proj", "o_proj"]`

## 📚 Dataset

- **Name:** Ancient-Indian-Wisdom ([Hugging Face Hub](https://huggingface.co/Abhaykoul/Ancient-Indian-Wisdom))
- **Type:** Instruction-Response pairs
- **Formatting:**
  ```
  ### Instruction:
  <instruction_text>

  ### Response:
  <response_text>
  ```

## ⚙️ Model & Tokenizer

- **Base Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Tokenizer:** Same as the base model

## 🏋️ Training Configuration

- **Frameworks:** `transformers`, `datasets`, `trl`, `peft`, `bitsandbytes`
- **Hyperparameters:**
  - `num_train_epochs = 200`
  - `per_device_train_batch_size = 4`
  - `gradient_accumulation_steps = 4`
  - `learning_rate = 1e-4`
  - `fp16 = True` (mixed precision for efficiency)
  - `evaluation_strategy = "steps"` (evaluate every 100 steps)
  - `save_steps = 100` (checkpoint every 100 steps)
  - `load_best_model_at_end = True`
  - **Optimizer:** `adamw_torch`

## 💾 Saving & Inference

- **Saving Path:** `fine-tuned-deepseek-r1-1.5b`
- **Inference Script:**
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  import torch

  model_path = "fine-tuned-deepseek-r1-1.5b"
  model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
  tokenizer = AutoTokenizer.from_pretrained(model_path)

  def generate_text(prompt, max_new_tokens=1000):
      inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
      with torch.no_grad():
          output = model.generate(
              **inputs,
              max_new_tokens=max_new_tokens,
              do_sample=True,
              temperature=0.5,
              top_k=50,
              top_p=0.9
          )
      return tokenizer.decode(output[0], skip_special_tokens=True)

  # Example Usage
  prompt = "In Yoga philosophy, what is the significance of ahimsa (non-violence)?"
  print(generate_text(prompt))
  ```

## 📦 Key Libraries

- `transformers`
- `datasets`
- `trl`
- `peft`
- `bitsandbytes`

---

### 🚀 **Contributions Welcome!**
Feel free to open issues, fork the repo, and submit pull requests to improve the fine-tuning pipeline or expand it to other domains.

### 📧 **License & Contact**
Distributed under the MIT License. For questions or collaborations, please reach out!

