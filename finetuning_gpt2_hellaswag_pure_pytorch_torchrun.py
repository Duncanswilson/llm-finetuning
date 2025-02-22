import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


# 1. Define a custom dataset
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

init_process_group(backend="nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# 2. Load and prepare data
dataset = load_dataset("hellaswag")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

train_dataset = HellaSwagDataset(dataset['train'], tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, pin_memory=True, sampler=DistributedSampler(train_dataset))

# 3. Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.train()

# 4. Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

# 5. Training loop
gpu_id = int(os.environ["LOCAL_RANK"])
model.to(gpu_id)
model = DDP(model, device_ids=[gpu_id])

num_epochs = 3
# from torch.profiler import profile, record_function, ProfilerActivity
for epoch in range(num_epochs):
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch['input_ids'].to(gpu_id)
        attention_mask = batch['attention_mask'].to(gpu_id)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # prof.export_chrome_trace(f"{gpu_id}-trace.json")
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# 6. Save the model
torch.save(model.state_dict(), 'gpt2_finetuned.pth')
destroy_process_group()


# 7. Inference
def generate_text(prompt, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
prompt = "Context: The man got a loan to buy a house. The next thing he did was"
generated_text = generate_text(prompt)
print(generated_text)