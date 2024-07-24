import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import gather_object

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

# 2. Load and prepare data
dataset = load_dataset("hellaswag")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

train_dataset = HellaSwagDataset(dataset['train'], tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=32, pin_memory=True)

# 3. Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.train()

# dataloader_config = DataLoaderConfiguration(non_blocking=True)
# accelerator = Accelerator(dataloader_config=dataloader_config)
accelerator = Accelerator()

# 4. Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_loader
)

# 5. Training loop
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
device = accelerator.device

num_epochs = 3
# from torch.profiler import profile, record_function, ProfilerActivity
for epoch in range(num_epochs):
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:

    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):

        # if step >= 10:
        #     break
        # with record_function("training_step"):
        # with accelerator.accumulate(model):
        input_ids = batch['input_ids'] #.to(device)
        attention_mask = batch['attention_mask'] #.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        accelerator.backward(loss)
        optimizer.step()
        
        total_loss += loss.item()

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# 6. Save the model
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
accelerator.save(unwrapped_model.state_dict(), f"gpt2_finetuned_{epoch}.pth")

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