# llm-finetuning

this repo is an exploration of finetuning LMs on custom datasets (for starters, the hellaswag training dataset) 

to run the single GPU case, `cd` into the repo and then run 
```
python finetuning_gpt2_hellaswag_pure_pytorch.py
``` 
on a snigle 3090, I get about ~4 mins per epoch. 

to run the multi-GPU case, make sure you have `accelerate` installed and then run
```
accelerate launch parallel_finetuning_gpt2_hellaswag_pure_pytorch.py
```
annoyingly, even though all 3 3090s on my set up get used, I see about ~12 mins per epoch.  
