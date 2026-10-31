
下载数据集

`python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial --use-huggingface`

跑实验

`MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 python libero/lifelong/main.py seed=100 benchmark_name=LIBERO_SPATIAL policy=bc_transformer_policy lifelong=er`

句柄错误问题修复

`/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_032n_LIBERO/LIBERO/libero/lifelong/utils.py`

```python
    data = next(iter(tmp_loader))
    if hasattr(dataset, "sequence_dataset"):
        dataset.sequence_dataset.close_and_delete_hdf5_handle()
```

`/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_032n_LIBERO/LIBERO/libero/lifelong/datasets.py`

```python
    dataset.obs_modality = obs_modality
    return dataset, shape_meta
```

```python
    def __getitem__(self, idx):
        if ObsUtils.OBS_KEYS_TO_MODALITIES is None and hasattr(self.sequence_dataset, "obs_modality"):
            ObsUtils.initialize_obs_utils_with_obs_specs({"obs": self.sequence_dataset.obs_modality})
```

推荐启动命令

```bash
PYTHONWARNINGS=ignore MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
python libero/lifelong/main.py \
  seed=100 benchmark_name=LIBERO_SPATIAL \
  policy=bc_transformer_policy lifelong=er \
  2>&1 | tee ./_001n_my/_006n_logs/libero_spatial_er_seed100.log \
  | grep --line-buffered -E "Epoch|Task|succ|train time|Error|Traceback|KeyboardInterrupt"
```
