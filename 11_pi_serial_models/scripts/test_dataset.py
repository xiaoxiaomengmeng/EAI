# test_dataset.py
from lerobot.datasets.lerobot_dataset import LeRobotDataset   # v0.4.4：无 .common 子包

dataset = LeRobotDataset(
    repo_id="aloha_sim_transfer_cube_human",
    root="/home/xy/xmm_ws/pi0/data/aloha_sim_transfer_cube_human",   # 数据集目录本身（含 meta/info.json）
)
print("episodes:", dataset.num_episodes)
print("frames:", len(dataset))
sample = dataset[0]
print("keys:", list(sample.keys()))
print("action shape:", sample["action"].shape)
print("image shape:", sample["observation.images.top"].shape)

