#!/bin/bash
# 多数据集合并流水线: 对齐 feature (取交集, 删多余相机) -> aggregate 成单一训练数据集。
# 场景: 260701(notac, 无触觉相机) + 260702(有 4 个 finger 相机) 一起训。二者 feature 不一致,
#       aggregate 会因 "Same features is expected" 报错; 本脚本先删掉多出来的 finger 相机使其对齐,
#       再合并。结果是一个 notac 数据集 (wrist+top + state/action + EE 列), 用 tactile_mode=none 训练。
# 用法: bash merge_datasets.sh <out_id> <src_id_1> <src_id_2> [<src_id_3> ...]
#   所有 id 都是 playground/data 下的目录名 (不含路径)。
# 产物 (非破坏, 原始数据集不动):
#   playground/data/<out_id>   <-- 合并后, 训练用这个
# 注: 默认按 10MB 分片视频 (--video-files-size-in-mb), 保持与源数据集一致的小 mp4 布局;
#     大 mp4 会让训练随机取帧变慢 (pyav 每个 __getitem__ 都重新解析整个文件索引)。
set -e
cd "$(dirname "$0")/.." || exit 1   # 切到仓库根, 服务器/本地通用
REPO_ROOT="$(pwd)"

dataset_root=playground/data

out_id=waic_final

srcs=(
  rm_umi_dual_260706_pen_in_case_notac_undist_256
  rm_umi_dual_260706_pen_in_case_1_notac_undist_256
  rm_umi_dual_260707_pen_in_case_notac_undist_256
  rm_umi_dual_260707_pen_in_case_1_notac_undist_256
  rm_umi_dual_260708_pen_in_case_notac_undist_256
  rm_umi_dual_260711_pen_in_case_undist_256
  waic_6_undist_256
  waic_14_undist_256
)


roots=()
for s in "${srcs[@]}"; do
  roots+=("${dataset_root}/${s}")
done
out="${dataset_root}/${out_id}"

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}

echo "==================================================================="
echo "合并数据集 -> ${out}"
for s in "${srcs[@]}"; do echo "  源: ${dataset_root}/${s}"; done
echo "==================================================================="

if [ -d "${out}" ]; then
  echo "错误: 输出目录已存在, 拒绝覆盖: ${out}"; exit 1
fi

python tools/merge_datasets.py \
  --roots "${roots[@]}" \
  --out "${out}" \
  --repo-id "${out_id}"

echo "==================================================================="
echo "完成 ✅  训练数据集: ${out}"
echo "  训练示例: bash train.sh ${out_id} pi05 4 16 20000 false none episode_rot6d relative_rot6d"
echo "==================================================================="
