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


