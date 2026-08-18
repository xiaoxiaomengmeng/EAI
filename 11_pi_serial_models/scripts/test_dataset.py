# test_dataset.py
# 验证本机数据集可被 LeRobotDataset 正常加载（v0.6.2 实测通过：50 episodes / 20000 帧，v3.0 格式）
from lerobot.datasets.lerobot_dataset import LeRobotDataset   # v0.6.2：导入路径与 v0.4.4 相同

dataset = LeRobotDataset(
    repo_id="aloha_sim_transfer_cube_human",
    root="/home/creator/ws/11_pi_serial_models/data/aloha_sim_transfer_cube_human",   # 数据集目录本身（含 meta/info.json）
)
print("episodes:", dataset.num_episodes)          # 预期 50
print("frames:", len(dataset))                    # 预期 20000
sample = dataset[0]
print("keys:", list(sample.keys()))
print("action shape:", sample["action"].shape)                    # 预期 torch.Size([14])
print("image shape:", sample["observation.images.top"].shape)     # 预期 torch.Size([3, 480, 640])

