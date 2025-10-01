# src/models/components/echoprime_encoders.py

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import transformers


class EchoPrimeVideoEncoder(pl.LightningModule):
    """
    一个独立的视频编码器模块，严格遵循EchoPrime的实现。
    """

    def __init__(self, pretrained_path, frozen=True):
        super().__init__()
        self.model = torchvision.models.video.mvit_v2_s()
        self.model.head[-1] = nn.Linear(self.model.head[-1].in_features, 512)

        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        weights = {k.replace('model.', '').replace('echo_encoder.', ''): v for k, v in checkpoint.items()}
        self.model.load_state_dict(weights)
        print(f"EchoPrime Video Encoder weights loaded from: {pretrained_path}")

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)


class EchoPrimeTextEncoder(pl.LightningModule):
    """
    一个独立的文本编码器模块，严格遵循EchoPrime的实现。
    使用BiomedBERT作为骨干网络。
    """

    def __init__(self, pretrained_path, frozen=True):
        super().__init__()

        config = transformers.AutoConfig.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        # config = transformers.AutoConfig.from_dict(config_dict)

        # ** 核心修正点 1: 创建与权重文件匹配的MLM模型架构 **
        # 我们先创建一个完整的AutoModelForMaskedLM，因为它包含了权重文件中所有的键
        temp_model = transformers.AutoModelForMaskedLM.from_config(config)

        self.text_projection = nn.Linear(768, 512)

        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            # 重命名权重文件中的键以匹配temp_model的结构
            # EchoPrime权重中的 'bert' 层级对应于Hugging Face模型中的基础模型名（通常也是'bert'）
            renamed_checkpoint = {}
            for key, value in checkpoint.items():
                if key.startswith("backbone.bert."):
                    new_key = key.replace("backbone.bert.", "bert.")
                    renamed_checkpoint[new_key] = value
                elif key.startswith("backbone.cls."):
                    new_key = key.replace("backbone.cls.", "cls.")
                    renamed_checkpoint[new_key] = value
                else:
                    renamed_checkpoint[key] = value

            # 将重命名后的权重加载到MLM模型中
            temp_model.load_state_dict(renamed_checkpoint, strict=False)
            print(
                f"EchoPrime Text Encoder weights successfully loaded into temporary MLM model from: {pretrained_path}")

        # ** 核心修正点 2: "拆掉"MLM头，只保留我们需要的基础BERT模型 **
        # temp_model.bert 就是我们需要的、包含了正确权重的 "编码器 + 池化头"
        self.backbone = temp_model.bert

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        )

        # tokenizer_config_dict = {
        #     "do_lower_case": "true"
        # }
        # self.tokenizer = transformers.AutoTokenizer.from_dict(tokenizer_config_dict)



        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, text_prompts):
        tokens = self.tokenizer(
            text_prompts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # 使用self.backbone进行特征提取
        outputs = self.backbone(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        # 取[CLS] token的输出
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projected_embedding = self.text_projection(cls_embedding)
        return projected_embedding