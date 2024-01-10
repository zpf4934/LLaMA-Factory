# -*- coding:utf-8 -*-
"""
# File       : temp.py
# Time       ：2024/1/9 14:00
# Author     ：andy
# version    ：python 3.9
"""
import json
from tqdm import tqdm

lines = open('/aigc/dataclub/微调数据集/base_sft.jsonl').readlines()
with open('/aigc/dataclub/微调数据集/single_base_sft.jsonl', 'w') as fw, open('/aigc/dataclub/微调数据集/mult_base_sft.jsonl', 'w') as fw1:
    for line in tqdm(lines):
        line = json.loads(line)
        if line['instruction'] and line['output']:
            if 'history' in line:
                fw1.write(json.dumps(line, ensure_ascii=False) + '\n')
            else:
                fw.write(json.dumps(line, ensure_ascii=False) + '\n')


