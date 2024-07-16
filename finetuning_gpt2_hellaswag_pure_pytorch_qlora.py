import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BitsAndBytesConfig
from transformers.models.gpt2.modeling_gpt2 import Conv1D
from datasets import load_dataset
from tqdm import tqdm
import bitsandbytes as bnb

# LoRA implementation
class QLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=32):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lora_A = bnb.nn.Linear4bit(
            in_features, rank, bias=False,
            compress_statistics=True,
            quant_type="nf4"
        ).to(device)
        self.lora_B = bnb.nn.Linear4bit(
            rank, out_features, bias=False,
            compress_statistics=True,
            quant_type="nf4"
        ).to(device)
        self.scaling = alpha / rank
        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self):
        # For Linear4bit, we can't directly access or set the weight
        # Instead, we can use the underlying FP16 weights for initialization
        nn.init.kaiming_uniform_(self.lora_A.weight.float(), a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight.float())

    def forward(self, x):
        intermediate = self.lora_A(x)
        result = self.lora_B(intermediate) * self.scaling
        return result

class LoRALinear4bit(nn.Module):
    def __init__(self, Linear4bit, rank=8, alpha=32):
        super().__init__()
        self.Linear4bit = Linear4bit
        in_features = Linear4bit.in_features  
        out_features = Linear4bit.out_features
        self.qlora = QLoRALayer(in_features, out_features, rank, alpha)

    def forward(self, x):
        conv_output = self.Linear4bit(x)
        lora_output = self.qlora(x)
        return conv_output + lora_output

def add_lora_to_model(model, rank=8, alpha=32):
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit) and 'lm_head' not in name:
            parent_name = '.'.join(name.split('.')[:-1])
            module_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name)
            lora_layer = LoRALinear4bit(module, rank, alpha)
            setattr(parent, module_name, lora_layer)
            lora_params.extend(lora_layer.qlora.lora_A.parameters())
            lora_params.extend(lora_layer.qlora.lora_B.parameters())
    
    print(f"Total LoRA parameters added: {sum(p.numel() for p in lora_params)}")
    return lora_params
    
    print(f"Total LoRA parameters added: {lora_params_count}")

class HellaSwagDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['ctx']
        correct_ending = item['endings'][int(item['label'])]
        text = f"Context: {context} Ending: {correct_ending}"
        
        encoding = self.tokenizer(text, truncation=True, padding='max_length', 
                                  max_length=self.max_length, return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

# Load and prepare data
dataset = load_dataset("hellaswag")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

train_dataset = HellaSwagDataset(dataset['train'], tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Load model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = GPT2LMHeadModel.from_pretrained(
    'gpt2',
    quantization_config=quantization_config,
    device_map="auto"
)
model.enable_input_require_grads()
# Add LoRA layers and get the names of LoRA parameters
lora_param_names = add_lora_to_model(model)
model.train()

# After adding LoRA layers, freeze the original parameters
# Freeze original parameters and identify LoRA parameters
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        lora_params.append(param)
        #print(f"Trainable LoRA parameter: {name}, shape: {param.shape}")
    else:
        param.requires_grad = False

# Only optimize LoRA parameters
# Create optimizer only for LoRA parameters
optimizer = bnb.optim.PagedAdamW8bit(lora_params, lr=5e-5)

# Debug prints
print(f"LoRA parameters found for optimizer: {len(lora_params)}")
print(f"Total LoRA parameter count: {sum(p.numel() for p in lora_params)}")


# Training loop
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

num_epochs = 3
# from torch.profiler import profile, record_function, ProfilerActivity
for epoch in range(num_epochs):
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            # if step >= 10:
            #     break
            # with record_function("training_step"):

        input_ids = batch['input_ids'] #.to(device)
        attention_mask = batch['attention_mask'] #.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'gpt2_finetuned.pth')

# Inference
def generate_text(prompt, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt') #.to(device)
    
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "Context: The man got a loan to buy a house. The next thing he did was"
generated_text = generate_text(prompt)
print(generated_text)