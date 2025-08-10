# training_script.py
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os

# Import your model and related classes
from your_module import (
    BartWithRegression, 
    Trainer, 
    TrainingArguments, 
    MultiTaskBartDataCollator,
    setup_distributed,
    cleanup_distributed,
    get_device_info
)
from multi_task_bart.losses import BartWithRegressionCriterion  # Your loss function

# ===========================
# OPTION 1: Single GPU Training
# ===========================
def train_single_gpu():
    """Simple single GPU or CPU training"""
    
    # Initialize model
    model = BartWithRegression(
        single_sep_token=True,
        regression_dropout=0.1,
        verbose=True
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create your datasets and dataloaders
    # train_dataset = YourDataset(...)  # Your custom dataset
    # eval_dataset = YourDataset(...)
    # test_dataset = YourDataset(...)
    
    tokenizer = BartWithRegression.get_tokenizer(single_sep_token=True)
    data_collator = MultiTaskBartDataCollator(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=data_collator)
    # eval_dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)
    # test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)
    
    # Setup training arguments
    criterion = BartWithRegressionCriterion(reg_weight=1.0, gen_weight=1.0)
    
    training_args = TrainingArguments(
        criterion=criterion,
        num_epochs=3,
        gradient_accumulation_steps=4,
        body_lr=3e-5,
        head_lr=1.5e-4,
        weight_decay=0.01,
        early_stopping_patience=3,
        save_path="./model_checkpoint"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        args=training_args,
        # train_dataloader=train_dataloader,
        # eval_dataloader=eval_dataloader,
        # test_dataloader=test_dataloader
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate
    # results = trainer.evaluate()
    # print("Evaluation results:", results)

# ===========================
# OPTION 2: DataParallel (Simple Multi-GPU)
# ===========================
def train_data_parallel():
    """Simple multi-GPU training using DataParallel"""
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("DataParallel requires multiple GPUs")
        return train_single_gpu()
    
    print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    
    # Initialize model
    model = BartWithRegression(
        single_sep_token=True,
        regression_dropout=0.1,
        verbose=True
    )
    
    device = torch.device('cuda')
    
    # The Trainer class will automatically wrap with DataParallel if multiple GPUs detected
    training_args = TrainingArguments(
        criterion=BartWithRegressionCriterion(reg_weight=1.0, gen_weight=1.0),
        num_epochs=3,
        gradient_accumulation_steps=4,
        body_lr=3e-5,
        head_lr=1.5e-4,
        weight_decay=0.01,
        early_stopping_patience=3,
        save_path="./model_checkpoint",
        use_distributed=False  # This will use DataParallel instead
    )
    
    trainer = Trainer(
        model=model,
        device=device,
        args=training_args,
        # Add your dataloaders here
    )
    
    trainer.train()

# ===========================
# OPTION 3: Distributed Training (Most Efficient)
# ===========================
def train_distributed_worker(rank, world_size, train_dataset, eval_dataset, test_dataset):
    """Worker function for distributed training"""
    
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Initialize model
    model = BartWithRegression(
        single_sep_token=True,
        regression_dropout=0.1,
        verbose=(rank == 0)  # Only print on main process
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    eval_sampler = DistributedSampler(
        eval_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    )
    test_sampler = DistributedSampler(
        test_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    )
    
    # Create dataloaders with distributed samplers
    tokenizer = BartWithRegression.get_tokenizer(single_sep_token=True)
    data_collator = MultiTaskBartDataCollator(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=8,  # Per-GPU batch size
        sampler=train_sampler,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=8,
        sampler=eval_sampler,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=8,
        sampler=test_sampler,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4
    )
    
    # Setup training arguments for distributed training
    training_args = TrainingArguments(
        criterion=BartWithRegressionCriterion(reg_weight=1.0, gen_weight=1.0),
        num_epochs=3,
        gradient_accumulation_steps=4,
        body_lr=3e-5,
        head_lr=1.5e-4,
        weight_decay=0.01,
        early_stopping_patience=3,
        save_path="./model_checkpoint" if rank == 0 else None,  # Only save on main process
        use_distributed=True,
        local_rank=rank,
        find_unused_parameters=False
    )
    
    device = torch.device(f'cuda:{rank}')
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        test_dataloader=test_dataloader
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate (only on main process)
    if rank == 0:
        results = trainer.evaluate()
        print("Evaluation results:", results)
    
    # Cleanup
    cleanup_distributed()

def train_distributed():
    """Launch distributed training"""
    
    if not torch.cuda.is_available():
        print("Distributed training requires CUDA")
        return train_single_gpu()
    
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Distributed training requires multiple GPUs")
        return train_data_parallel()
    
    print(f"Starting distributed training with {world_size} GPUs")
    
    # Load your datasets here
    # train_dataset = YourDataset(...)
    # eval_dataset = YourDataset(...)
    # test_dataset = YourDataset(...)
    
    # This would be uncommented when you have actual datasets:
    mp.spawn(
        train_distributed_worker,
        args=(world_size, train_dataset, eval_dataset, test_dataset),
        nprocs=world_size,
        join=True
    )

# ===========================
# OPTION 4: Using torchrun (Recommended for Production)
# ===========================
def train_with_torchrun():
    """Training function for use with torchrun command"""
    
    # Get distributed training info from environment variables
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    if world_size > 1:
        # Initialize distributed training
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    # Initialize model
    model = BartWithRegression(
        single_sep_token=True,
        regression_dropout=0.1,
        verbose=(rank == 0)
    )
    
    # Setup datasets and dataloaders with distributed samplers if needed
    # ... (similar to train_distributed_worker)
    
    # Setup training arguments
    training_args = TrainingArguments(
        criterion=BartWithRegressionCriterion(reg_weight=1.0, gen_weight=1.0),
        num_epochs=3,
        gradient_accumulation_steps=4,
        body_lr=3e-5,
        head_lr=1.5e-4,
        weight_decay=0.01,
        early_stopping_patience=3,
        save_path="./model_checkpoint" if rank == 0 else None,
        use_distributed=(world_size > 1),
        local_rank=local_rank,
        find_unused_parameters=False
    )
    
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        args=training_args,
        # Add your dataloaders here
    )
    
    # Train the model
    trainer.train()
    
    # Cleanup distributed training
    if world_size > 1:
        torch.distributed.destroy_process_group()

# ===========================
# Main execution
# ===========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_mode', type=str, default='auto', 
                       choices=['single', 'parallel', 'distributed', 'torchrun', 'auto'])
    args = parser.parse_args()
    
    if args.training_mode == 'single':
        train_single_gpu()
    elif args.training_mode == 'parallel':
        train_data_parallel()
    elif args.training_mode == 'distributed':
        train_distributed()
    elif args.training_mode == 'torchrun':
        train_with_torchrun()
    else:  # auto
        device, num_gpus, strategy = get_device_info()
        print(f"Auto-detected: {strategy} with {num_gpus} device(s)")
        
        if strategy == 'single':
            train_single_gpu()
        elif strategy == 'parallel':
            train_data_parallel()