import os
import torch
import torch.distributed as dist

def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if local_rank >= n:
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but torch.cuda.device_count()={n}. "
                    f"Fix nproc_per_node or CUDA_VISIBLE_DEVICES."
                )
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, dist.get_rank(), local_rank, dist.get_world_size()
    return False, 0, 0, 1

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def ddp_barrier(ddp_enabled: bool):
    if ddp_enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()

def ddp_concat_all_gather(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    world = dist.get_world_size()
    device = x.device

    n_local = torch.tensor([x.shape[0]], device=device, dtype=torch.long)
    n_list = [torch.zeros_like(n_local) for _ in range(world)]
    dist.all_gather(n_list, n_local)
    n_list = [int(t.item()) for t in n_list]
    n_max = max(n_list)

    if x.shape[0] < n_max:
        pad = torch.zeros((n_max - x.shape[0],) + x.shape[1:], device=device, dtype=x.dtype)
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x

    gather_list = [torch.zeros_like(x_pad) for _ in range(world)]
    dist.all_gather(gather_list, x_pad)

    out = []
    for gi, ni in zip(gather_list, n_list):
        out.append(gi[:ni])
    return torch.cat(out, dim=0)
