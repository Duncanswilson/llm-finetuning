# llm-finetuning

this repo is an exploration of finetuning LMs on custom datasets (for starters, finetuning GPT2 on the hellaswag training dataset) 

we've fixed the local bug that was causing slowdowns (it was a BIOS level change of the PCI Link speed) so now these scrip  ts are performing and show speedup when run across multiple GPUs. 

recently added a DPO script which leverages `peft` and `trl` to do Direct Preference Optimization on Intel's version of the ORCA dataset.