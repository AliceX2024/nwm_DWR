# test_sync.py - 快速验证同步修复
import torch
import torch.distributed as dist
from distributed import init_distributed
import time

def test_sync():
    _, rank, device, _ = init_distributed()
    
    print(f"Rank {rank} started")
    
    # 模拟训练循环中的同步点
    for i in range(10):
        # 模拟一些计算
        time.sleep(0.1)
        
        # 测试日志打印时的同步（关键测试点）
        if i % 3 == 0:  # 模拟 log_every
            dist.barrier()  # 这是你添加的修复
            print(f"Rank {rank} passed barrier at iter {i}")
            
            # 测试 all_reduce
            test_tensor = torch.tensor(float(rank), device=device)
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            print(f"Rank {rank} completed all_reduce at iter {i}, result: {test_tensor.item()}")
    
    # 测试 checkpoint 保存时的同步
    dist.barrier()
    if rank == 0:
        print("Rank 0 saving checkpoint...")
        time.sleep(0.5)  # 模拟保存时间
    dist.barrier()
    print(f"Rank {rank} passed checkpoint barrier")
    
    print(f"Rank {rank} test completed successfully!")
    dist.destroy_process_group()

if __name__ == "__main__":
    test_sync()