import os
import pathlib
import torch
import glob

def save_checkpoint(model, checkpoint_dir, step, optimizer, keep_latest=3, lr_scheduler=None):
    model_name = "model-%08d.pth"%(step)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    prev_chkpts = list(pathlib.Path(checkpoint_dir).glob('model-*'))
    prev_chkpts.sort(key=lambda p: p.stat().st_mtime,reverse=True)
    if len(prev_chkpts) > keep_latest-1:
        for f in prev_chkpts[keep_latest-1:]:
            f.unlink()
    path = os.path.join(checkpoint_dir, model_name)
    if optimizer is None and lr_scheduler is None:
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            }, path)
    elif lr_scheduler is None:
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, path)
    else:
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, path)
    print("Saved a checkpoint: %s"%(path))



def load(model_name, checkpoint_root, model, optimizer, lr_scheduler=None, strict=True):
    print("reading full checkpoint...")
    checkpoint_dir = os.path.join(checkpoint_root, model_name)
    step = 0
    if not os.path.exists(checkpoint_dir):
        print("...ain't no full checkpoint here!")
        print(checkpoint_dir)
        assert(False)
    else:
        ckpt_names = os.listdir(checkpoint_dir)
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            step = max(steps)
            model_name = 'model-%08d.pth' % (step)
            path = os.path.join(checkpoint_dir, model_name)
            print("...found checkpoint %s"%(path))

            checkpoint = torch.load(path)
            
            # # Print model's state_dict
            # print("Model's state_dict:")
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            # input()

            # # Print optimizer's state_dict
            # print("Optimizer's state_dict:")
            # for var_name in optimizer.state_dict():
            #     print(var_name, "\t", optimizer.state_dict()[var_name])
            # input()
            
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if lr_scheduler is not None:
                if 'lr_scheduler_state_dict' in checkpoint.keys():
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                else:
                    "WANRNING: LR SCHEDULER NOT IN CHECKPOINT. Returning lr_scheduler without loading state dict."
        else:
            print("...ain't no full checkpoint here!")
            print(checkpoint_dir)
            assert(False)
    return step


def load_from_path(path, model, optimizer, lr_scheduler=None, strict=True):
    print("reading full checkpoint...")
    step = 0
    # path = args.load_model_path 

    steps = int((path.split('-')[1]).split('.')[0])

    # if args.lr_scheduler_from_scratch:
    #     print("LR SCHEDULER FROM SCRATCH")
    #     lr_scheduler_load = False
    # else:
    #     lr_scheduler_load = True

    # if args.optimizer_from_scratch:
    #     print("OPTIMIZER FROM SCRATCH")
    #     optimizer_load = False
    # else:
    #     optimizer_load = True

    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if lr_scheduler is not None:
        if 'lr_scheduler_state_dict' in checkpoint.keys():
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        else:
            "WANRNING: LR SCHEDULER NOT IN CHECKPOINT. Returning lr_scheduler without loading state dict."
    print(f"Loaded {path}")
    return step