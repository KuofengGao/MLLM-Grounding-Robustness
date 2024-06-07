import os
os.environ['TORCH_HOME'] = '/your/path/models'
import re
import json
import argparse
from collections import defaultdict
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU
from minigpt4.conversation.conversation import CONV_VISION_minigptv2

from minigpt4.datasets.datasets.coco_caption import RefCOCOEvalData, RefCOCOEvalDataText

def list_of_str(arg):
    return list(map(str, arg.split(',')))

def clamp(delta, clean_imgs):
    # Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    MEAN = torch.tensor([[[0.48145466]], [[0.4578275]], [[0.40821073]]]).to(clean_imgs.device)
    STD = torch.tensor([[[0.26862954]], [[0.26130258]], [[0.27577711]]]).to(clean_imgs.device)

    clamp_imgs = (((delta.data + clean_imgs.data) * STD + MEAN) * 255).clamp(0, 255)
    clamp_delta = (clamp_imgs/255 - MEAN) / STD - clean_imgs.data

    return clamp_delta

parser = eval_parser()
parser.add_argument("--dataset", type=str, default='refcoco', help="dataset to evaluate")
parser.add_argument("--split", type=str, default='val', help="dataset to split")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")
parser.add_argument("--resample", action='store_true', help="resolution used in refcoco")
parser.add_argument('--epsilon', type=float, default=0.064, help='the epsilon of the adversarial attack')
parser.add_argument('--step_size', type=float, default=0.0078, help='the step size of the adversarial attack')
parser.add_argument('--iter', type=int, default=100, help='the iteration of the adversarial attack')
args = parser.parse_args()

cfg = Config(args)

eval_dict = {'refcoco': ['val','testA','testB'], 
            'refcoco+': ['val','testA','testB'],
            'refcocog': ['val','test']}


model, vis_processor = init_model(args)
model.eval()
for name, param in model.named_parameters():
    param.requires_grad = False
CONV_VISION = CONV_VISION_minigptv2
conv_temp = CONV_VISION.copy()
conv_temp.system = ""

# 
save_path = cfg.run_cfg.save_path



# for dataset in args.dataset:
#     for split in eval_dict[dataset]:
dataset = args.dataset
split = args.split
eval_file_path = cfg.evaluation_datasets_cfg[dataset]["eval_file_path"]
img_path = cfg.evaluation_datasets_cfg[dataset]["img_path"]
batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]

with open(os.path.join(eval_file_path,f"{args.dataset}_{args.split}.json"), 'r') as f:
    refcoco = json.load(f)

data = RefCOCOEvalDataText(refcoco, vis_processor, img_path)
eval_dataloader = DataLoader(data, batch_size=2, shuffle=False)
minigpt4_predict = defaultdict(list)
resamples = []
criterion = torch.nn.MSELoss()
STEP_SIZE, EPSILON, ITER = args.step_size, args.epsilon, args.iter

for images, questions, img_ids, answers in tqdm(eval_dataloader):
    images = images.half().cuda()
    delta = torch.rand_like(images, requires_grad=True).half().cuda()
    item = {}
    item["instruction_input"] = ["<Img><ImageHere></Img> {}".format(question) for question in questions]
    item["answer"] = ['{<0><0><20><20>}' for _ in range(int(images.shape[0]))]
    for it in range(ITER):
        item["image"] = images + delta
        loss = model.forward(item)['loss']     
        model.zero_grad()
        loss.backward(retain_graph=False)
        delta.data = delta - STEP_SIZE * torch.sign(delta.grad.detach())
        delta.data = clamp(delta, images).clamp(-EPSILON, EPSILON)
        delta.grad.zero_()

    
    
    texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
    answers = model.generate(images + delta, texts, max_new_tokens=max_new_tokens, do_sample=False)
    for answer, img_id, question in zip(answers, img_ids, questions):
        answer = answer.replace("<unk>","").replace(" ","").strip()
        pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
        if re.match(pattern, answer):
            minigpt4_predict[img_id].append(answer)
        else:
            resamples.append({'img_id': img_id, 'sents': [question.replace('[refer] give me the location of','').strip()]})

    
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
file_save_path = os.path.join(save_path,f"{args.dataset}_{split}.json")
with open(file_save_path,'w') as f:
    json.dump(minigpt4_predict, f)

count=0
total=len(refcoco)
res=args.res
refcoco_dict = defaultdict()
for item in refcoco:
    refcoco_dict[item['img_id']] = item
for img_id in refcoco_dict:
    item = refcoco_dict[img_id]
    bbox = item['bbox']
    outputs = minigpt4_predict[img_id]
    for output in outputs:
        try:
            integers = re.findall(r'\d+', output)
            pred_bbox = [int(num) for num in integers]
            height = item['height']
            width = item['width']
            pred_bbox[0] = pred_bbox[0] / res * width
            pred_bbox[1] = pred_bbox[1] / res * height
            pred_bbox[2] = pred_bbox[2] / res * width
            pred_bbox[3] = pred_bbox[3] / res * height

            gt_bbox = [0,0,0,0]
            gt_bbox[0] = 0 # bbox[0]
            gt_bbox[1] = 0 # bbox[1]
            gt_bbox[2] = width / 5 # bbox[0] + bbox[2]
            gt_bbox[3] = height / 5 # bbox[1] + bbox[3]

            iou_score = computeIoU(pred_bbox, gt_bbox)
            if iou_score > 0.5:
                count+=1
        except:
            continue

print(f'{dataset} {split}:', count / total * 100, flush=True)
minigpt4_predict['acc'] = count / total * 100
with open(file_save_path,'w') as f:
    json.dump(minigpt4_predict, f)
