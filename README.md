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

mimamba_echo_prime_text_video

训练camus数据集
```angular2html
nohup python train.py --config-name train_mimamba_echo_prime_text_video_camus >> output1.log 2>&1 &
```