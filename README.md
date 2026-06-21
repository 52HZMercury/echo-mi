# EchoMI


```angular2html
nohup python train.py --config-name train_mimamba_camus >> output1.log 2>&1 &
```

多切面训练

简单多切面
mimamba_double_view
```angular2html
nohup python train.py --config-name train_mimamba_double_view_camus >> output1.log 2>&1 &
```

分层融合模型
mimamba_hierarchical
```angular2html
nohup python train.py --config-name train_mimamba_hierarchical_camus >> output1.log 2>&1 &
```

mimamba_cross_ssm模型

训练camus数据集
```angular2html
nohup python train.py --config-name train_mimamba_cross_ssm_camus >> output1.log 2>&1 &
```

训练hmc数据集
```angular2html
nohup python train.py --config-name train_mimamba_cross_ssm_hmc >> output1.log 2>&1 &
```

mimamba_echo_prime_video

训练camus数据集
```angular2html
nohup python train.py --config-name train_mimamba_echo_prime_video_camus >> output1.log 2>&1 &
```


训练省医院数据集
```angular2html
nohup python train.py --config-name train_mimamba_echo_prime_text_video_provincial_hospital >> output1.log 2>&1 &
```

省医院6切面数据集  mamba
```angular2html
nohup python train.py --config-name train_mimamba_6C_provincial_hospital >> output1.log 2>&1 &
```

省医院6切面数据集  echoprime
```angular2html
nohup python train.py --config-name train_echo_prime_6C_provincial_hospital >> output1.log 2>&1 &
```

省医院6切面数据集 3分类 mamba
```angular2html
nohup python train.py --config-name train_mimamba_6C_3_provincial_hospital >> output1.log 2>&1 &
```


mimamba_originmutimodel

```angular2html
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

训练camus数据集
```angular2html
nohup python train.py --config-name train_mimamba_originmutimodel_camus >> output1.log 2>&1 &
```

训练hmc数据集
```angular2html
nohup python train.py --config-name train_mimamba_originmutimodel_hmc >> output1.log 2>&1 &
```

mimamba_echo_prime_text_video

训练camus数据集
```angular2html
nohup python train.py --config-name train_mimamba_echo_prime_text_video_camus >> output1.log 2>&1 &
```

训练hmc数据集
```angular2html
nohup python train.py --config-name train_mimamba_echo_prime_text_video_hmc >> output1.log 2>&1 &
```

logvmamba
训练camus数据集
```angular2html
nohup python train.py --config-name train_logvmamba_camus >> output1.log 2>&1 &
```

训练hmc数据集
```angular2html
nohup python train.py --config-name train_logvmamba_hmc >> output1.log 2>&1 &
```


lkmunet
训练camus数据集
```angular2html
nohup python train.py --config-name train_lkmunet_camus >> output1.log 2>&1 &
```

训练hmc数据集
```angular2html
nohup python train.py --config-name train_lkmunet_hmc >> output1.log 2>&1 &
```

ukan3d
训练camus数据集
```angular2html
nohup python train.py --config-name train_ukan3d_camus >> output1.log 2>&1 &
```

训练hmc数据集
```angular2html
nohup python train.py --config-name train_ukan3d_hmc >> output1.log 2>&1 &
```


BI-Mamba
训练camus数据集
```angular2html
nohup python train.py --config-name train_BI_mamba_camus >> output1.log 2>&1 &
```

训练hmc数据集
```angular2html
nohup python train.py --config-name train_BI_mamba_hmc >> output1.log 2>&1 &
```

BI-Mamba_plugin
训练camus数据集
```angular2html
nohup python train.py --config-name train_BI_mamba_plugin_camus > output1.log 2>&1 &
```

训练hmc数据集
```angular2html
nohup python train.py --config-name train_BI_mamba_plugin_hmc > output1.log 2>&1 &
```


xfmamba
```angular2html
nohup python train.py --config-name train_xfmamba_camus > output1.log 2>&1 &
python train.py --config-name train_xfmamba_hmc
nohup python train.py --config-name train_xfmamba_plugin_camus > output1.log 2>&1 &
python train.py --config-name train_xfmamba_plugin_hmc
```



E_vim3
```angular2html
nohup python train.py --config-name train_E_vim3_camus > output1.log 2>&1 &
nohup python train.py --config-name train_E_vim3_hmc > output1.log 2>&1 &

nohup python train.py --config-name train_E_vim3_plugin_camus > output1.log 2>&1 &
nohup python train.py --config-name train_E_vim3_plugin_hmc > output1.log 2>&1 &

```

8/16/32-frame experiments
```bash
python train.py --config-name train_E_vim3_camus data.num_frames=8
python train.py --config-name train_E_vim3_camus data.num_frames=16
python train.py --config-name train_E_vim3_camus data.num_frames=32

# The same override works for BI-Mamba, XFMamba, plugin variants, and HMC.
python train.py --config-name train_BI_mamba_plugin_camus data.num_frames=32
python train.py --config-name train_xfmamba_plugin_hmc data.num_frames=8
```

`data.num_frames` only accepts `8`, `16`, or `32`. The selected length is used
by the train, validation, and test datasets. For plugin variants, the main
backbone uses the selected length while the frozen EchoPrime encoder internally
resamples its input to its native 16-frame length.

For a valid 32-frame experiment, do not expand an existing 16-frame `.pt`
cache. Generate a separate cache from the raw AVI files, then point the training
configuration to it:
```bash
python scripts/preprocess_camus.py --source_dir /path/to/CAMUS/avi \
  --dest_dir /path/to/CAMUS/pt_data_32 --num_frames 32
python train.py --config-name train_E_vim3_camus data.num_frames=32 \
  data.data_dir=/path/to/CAMUS/pt_data_32
```
