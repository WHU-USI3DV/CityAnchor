import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.CityAnchor import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

from lib.config import CONF
from data.sensaturban.model_util_sensaturban import SensatUrbanDatasetConfig
# import open3d as o3d
# import gradio as gr
import plotly.graph_objects as go
import json
import rasterio
from PIL import Image
import random
import copy
import re
import time
from collections import defaultdict

from model.LISA_ROI import LISAForCausalLM as LISAForCausalLM_ROI
from model.llava_ROI import conversation as conversation_lib
from model.llava_ROI.mm_utils import tokenizer_image_token as tokenizer_image_token_ROI

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    # For stage 1
    parser.add_argument("--version_stage_1", default="./LISA-13B-llama2-v1")
    parser.add_argument(
        "--precision_stage_1",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference in stage 1",
    )
    parser.add_argument("--image_size_stage_1", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length_stage_1", default=512, type=int)
    parser.add_argument("--load_in_8bit_stage_1", action="store_true", default=False)
    parser.add_argument("--load_in_4bit_stage_1", action="store_true", default=False)
    parser.add_argument(
        "--conv_type_stage_1",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    # For stage 1
    parser.add_argument("--version", default="./LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model for ROI (stage 1)
    global tokenizer_ROI
    tokenizer_ROI = AutoTokenizer.from_pretrained(
        args.version_stage_1,
        cache_dir=None,
        model_max_length=args.model_max_length_stage_1,
        padding_side="right",
        use_fast=False,
    )
    tokenizer_ROI.pad_token = tokenizer_ROI.unk_token
    args.seg_token_idx_ROI = tokenizer_ROI("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype_ROI = torch.float32
    if args.precision_stage_1 == "bf16":
        torch_dtype_ROI = torch.bfloat16
    elif args.precision_stage_1 == "fp16":
        torch_dtype_ROI = torch.half
    
    kwargs_ROI = {"torch_dtype": torch_dtype_ROI}
    if args.load_in_4bit_stage_1:
        kwargs_ROI.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit_stage_1:
        kwargs_ROI.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model_ROI = LISAForCausalLM_ROI.from_pretrained(
        args.version_stage_1, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx_ROI, **kwargs_ROI
    )

    model_ROI.config.eos_token_id = tokenizer_ROI.eos_token_id
    model_ROI.config.bos_token_id = tokenizer_ROI.bos_token_id
    model_ROI.config.pad_token_id = tokenizer_ROI.pad_token_id

    model_ROI.get_model().initialize_vision_modules(model_ROI.get_model().config)
    vision_tower_ROI = model_ROI.get_model().get_vision_tower()
    vision_tower_ROI.to(dtype=torch_dtype_ROI)

    if args.precision_stage_1 == "bf16":
        model_ROI = model_ROI.bfloat16().cuda()
    elif (
        args.precision_stage_1 == "fp16" and (not args.load_in_4bit_stage_1) and (not args.load_in_8bit_stage_1)
    ):
        vision_tower_ROI = model_ROI.get_model().get_vision_tower()
        model_ROI.model.vision_tower = None
        import deepspeed

        model_engine_ROI = deepspeed.init_inference(
            model=model_ROI,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model_ROI = model_engine_ROI.module
        model_ROI.model.vision_tower = vision_tower_ROI.half().cuda()
    elif args.precision_stage_1 == "fp32":
        model_ROI = model_ROI.float().cuda()
    
    vision_tower_ROI = model_ROI.get_model().get_vision_tower()
    vision_tower_ROI.to(device=args.local_rank)
    clip_image_processor_ROI = CLIPImageProcessor.from_pretrained(model_ROI.config.vision_tower)
    transform_ROI = ResizeLongestSide(args.image_size)
    model_ROI.eval() 

    # Create model for grounding (stage 2)
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    ## Loading val dataset for ROI forming and Target Grounding
    from lib.config import CONF
    from data.sensaturban.model_util_sensaturban import SensatUrbanDatasetConfig
    DC = SensatUrbanDatasetConfig()            
    print("preparing data...")
    import json
    SCANREFER_TRAIN = json.load(open(os.path.join("./CityAnchor_Release_Data/meta_data/WHU-Refer_val.json"))) # Training samples
    SCANREFER_VAL = json.load(open("./CityAnchor_Release_Data/meta_data/WHU-Refer_val.json")) # Testing samples
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, -1)
    scanrefer = {"train": scanrefer_train, "val": scanrefer_val}
    scanrefer = scanrefer["val"]
    file_path = "./WHU-Refer_Val.txt" # Changing this to record grounding results
    iou_file_root = "./IoU_File/"

    info_path = './CityAnchor_Release_Data/data_whu/whu_pc_info_translated_full.json'
    bbox_path = './CityAnchor_Release_Data/data_whu/Whu_bbox.json'

    with open(info_path, 'r', encoding='utf-8') as f:
        info_data = json.load(f)

    with open(bbox_path, 'r', encoding='utf-8') as f:
        bbox_data = json.load(f)
    
    print("The total val sample is: ",len(scanrefer))

    threshold_ROI = -1
    K = 5

    for i in range(len(scanrefer)):

        idx = i
        print(idx)
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        object_id = int(scanrefer[idx]["object_id"])
        object_id_ori = object_id 
        ann_id = int(scanrefer[idx]["ann_id"])
        query = scanrefer[idx]["description"]
        local_image = np.zeros((3, 9, 9))

        raster_file = os.path.join("./CityAnchor_Release_Data/data_whu/Whu_grid_0.005.tif")
        pc_file = os.path.join("./CityAnchor_Release_Data/data_whu/fused_whu_anno_v2.txt")

        import rasterio
        with rasterio.open(raster_file) as src:
            image = src.read()
        image_with_rectangle = image[:, :, :]

        point_cloud = np.loadtxt(pc_file)

        bbox = bbox_data[str(object_id_ori)].get("bbox")
        pixel = bbox_data[str(object_id_ori)].get("pixel")

        name = info_data[str(object_id_ori)].get("name")
        shp_name = info_data[str(object_id_ori)].get("shp_name")

        x1 = pixel[0]
        y1 = pixel[1]
        x2 = pixel[2]
        y2 = pixel[3]

        X_pixel = int((x1 + x2) / 2)
        Y_pixel = int((y1 + y2) / 2)
        X_delta = int(x2 - x1)
        Y_delta = int(y2 - y1)

        instance_bboxes = []
        instance_bbox = []
        instance_labels = []

        input_text_ROI = query
        conv_stage_1 = conversation_lib.conv_templates[args.conv_type_stage_1].copy()
        conv_stage_1.messages = []
        prompt_stage_1 = DEFAULT_IMAGE_TOKEN + "\n" + input_text_ROI
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt_stage_1 = prompt_stage_1.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv_stage_1.append_message(conv_stage_1.roles[0], prompt_stage_1)
        conv_stage_1.append_message(conv_stage_1.roles[1], "")
        prompt_stage_1 = conv_stage_1.get_prompt()

        image_path = raster_file
        image_np_stage_1 = cv2.imread(image_path)
        image_np_stage_1 = cv2.cvtColor(image_np_stage_1, cv2.COLOR_BGR2RGB)
        original_size_list_stage_1 = [image_np_stage_1.shape[:2]]

        image_clip_stage_1 = (
            clip_image_processor_ROI.preprocess(image_np_stage_1, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        if args.precision_stage_1 == "bf16":
            image_clip_stage_1 = image_clip_stage_1.bfloat16()
        elif args.precision_stage_1 == "fp16":
            image_clip_stage_1 = image_clip_stage_1.half()
        else:
            image_clip_stage_1 = image_clip_stage_1.float()
    
        image_stage_1 = transform.apply_image(image_np_stage_1)
        resize_list_stage_1 = [image_stage_1.shape[:2]]

        image_stage_1 = (
            preprocess(torch.from_numpy(image_stage_1).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision_stage_1 == "bf16":
            image_stage_1 = image_stage_1.bfloat16()
        elif args.precision_stage_1 == "fp16":
            image_stage_1 = image_stage_1.half()
        else:
            image_stage_1 = image_stage_1.float()

        input_ids_ROI = tokenizer_image_token(prompt_stage_1, tokenizer_ROI, return_tensors="pt")
        input_ids_ROI = input_ids_ROI.unsqueeze(0).cuda()

        output_ids_ROI, pred_masks_ROI = model_ROI.evaluate(
            image_clip_stage_1,
            image_stage_1,
            input_ids_ROI,
            resize_list_stage_1,
            original_size_list_stage_1,
            max_new_tokens=512,
            tokenizer=tokenizer_ROI,
        )
        output_ids_ROI = output_ids_ROI[0][output_ids_ROI[0] != IMAGE_TOKEN_INDEX]
        text_output_ROI = tokenizer_ROI.decode(output_ids_ROI, skip_special_tokens=False)
        text_output_ROI = text_output_ROI.replace("\n", "").replace("  ", " ")
        pred_mask_ROI = pred_masks_ROI[0].detach().cpu().numpy()[0]

        pred_mask_ROI_standardized = (pred_mask_ROI - np.mean(pred_mask_ROI)) / np.std(pred_mask_ROI)
        min_standardized = np.min(pred_mask_ROI_standardized)
        max_standardized = np.max(pred_mask_ROI_standardized)
        pred_mask_ROI_scaled = 2 * (pred_mask_ROI_standardized - min_standardized) / (max_standardized - min_standardized) - 1
     
        heat_value = pred_mask_ROI[image.shape[1] - Y_pixel, X_pixel]
        heat_value_nor = pred_mask_ROI_scaled[image.shape[1] - Y_pixel, X_pixel]
        print(heat_value_nor)

        if heat_value_nor < threshold_ROI:
            with open(file_path, "a", encoding='utf-8') as file:
                file.write("object_id:" + str(object_id_ori) + ", ann_id:" + str(ann_id) + ", rank:999" + "\n" )
            continue

        # multi-feature embedding
        pc_feats = []
        
        cand_instance_ids = [int(k) for k in bbox_data.keys()]
        cand_instance_ids = [k for k in cand_instance_ids if k != object_id]
        cand_instance_ids = cand_instance_ids[:] 
        nearest_instance_ids = []

        if name is not None:
            prompt_object_category = " The class of this object belongs to " + shp_name + "The name of this object is: " + name + ". Please describe this object highlighted by the red box. "
        if name is None:
            prompt_object_category = " The class of this object belongs to " + shp_name + ". Please describe this object highlighted by the red box. "

        prompt = prompt_object_category + query + " Please output segmentation mask."
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        # Preparing for pre-trained pc_feats
        pc_feats = []
        feat_instance = np.zeros((1, 1024))
        pc_feats.append(feat_instance)
        pc_feats = np.vstack(pc_feats)
        pc_feats = pc_feats[0]

        # Preparing for landmark_feats
        landmark_feats = []
        landmark_feat_instance = np.zeros((1, 128))
        landmark_feats.append(landmark_feat_instance)
        landmark_feats = np.vstack(landmark_feats)
        landmark_feats = landmark_feats[0]

        # For nearest_pc
        try: 
          while len(nearest_instance_ids) < K: # K value 
            nearest_instance_ids.append(object_id)
        except AttributeError:
          nearest_instance_ids = []
          while len(nearest_instance_ids) < K: # K value 
            nearest_instance_ids.append(object_id)

        pc_feats_nearest = []
        for i in nearest_instance_ids:
            feat_instance = np.zeros((1, 1024))
            pc_feats_nearest.append(feat_instance)
        pc_feats_nearest = np.vstack(pc_feats_nearest)

        landmark_feats_nearest = []
        for i in nearest_instance_ids:
            landmark_feat_instance = np.zeros((1, 128))
            landmark_feats_nearest.append(landmark_feat_instance)
        landmark_feats_nearest = np.vstack(landmark_feats_nearest)
        
        # Preparing for image
        x1 = pixel[0]
        y1 = pixel[1]
        x2 = pixel[2]
        y2 = pixel[3]

        X_pixel = int((x1 + x2) / 2)
        Y_pixel = int((y1 + y2) / 2)
        X_delta = int(x2 - x1)
        Y_delta = int(y2 - y1)    
        thickness = min(6, x2 - x1, y2 - y1)

        if min(x2 - x1, y2 - y1) < 100:
            Image_LH = 5 * max(X_delta, Y_delta)
            thickness = 4
        
        if min(x2 - x1, y2 - y1) > 100:
            Image_LH = 6 * max(X_delta, Y_delta)
            thickness = 5

        for t in range(thickness):
          try: 
            image_with_rectangle[0, y1:y2, x1 + t + 1] = 255
            image_with_rectangle[0, y1:y2, x2 - t - 1] = 255
            image_with_rectangle[0, y1 + t + 1, x1:x2] = 255
            image_with_rectangle[0, y2 - t - 1, x1:x2] = 255
            image_with_rectangle[1, y1:y2, x1 + t + 1] = 0
            image_with_rectangle[1, y1:y2, x2 - t - 1] = 0
            image_with_rectangle[1, y1 + t + 1, x1:x2] = 0
            image_with_rectangle[1, y2 - t - 1, x1:x2] = 0
            image_with_rectangle[2, y1:y2, x1 + t + 1] = 0
            image_with_rectangle[2, y1:y2, x2 - t - 1] = 0
            image_with_rectangle[2, y1 + t + 1, x1:x2] = 0
            image_with_rectangle[2, y2 - t - 1, x1:x2] = 0
          except IndexError:
            continue
          
        if min(x2 - x1, y2 - y1) >= 100:
            local_image = extract_subimage(image_with_rectangle, image.shape[1] - Y_pixel, X_pixel, 5 * Y_delta, 5 * X_delta)
            
        if min(x2 - x1, y2 - y1) < 100:
            local_image = extract_subimage(image_with_rectangle, image.shape[1] - Y_pixel, X_pixel, 6 * Y_delta, 6 * X_delta)
            
        local_image = np.transpose(local_image, (1, 2, 0))
        image_np = local_image
        original_size_list = [image_np.shape[:2]]

        try:
            image_clip = (
                clip_image_processor.preprocess(image_np, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
                )
        except ZeroDivisionError:
            with open(file_path, "a", encoding='utf-8') as file:
                file.write("object_id:" + str(object_id_ori) + ", ann_id:" + str(ann_id) + ", rank:" + "999" + ", pos:" + "0" + "\n" )
            continue

        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()
        
        try:
            image = transform.apply_image(image_np)
        except ValueError:
            continue

        resize_list = [image.shape[:2]]
        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        lang_ans_object = model.evaluate(
            pc_feats,
            landmark_feats,
            pc_feats_nearest,
            landmark_feats_nearest,
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        
        print(lang_ans_object)
        lang_feat_list = []
        object_list = []
        print(cand_instance_ids)

        for i_2 in cand_instance_ids:
            conv = conversation_lib.conv_templates[args.conv_type].copy()
            conv.messages = []

            raster_file = os.path.join("./CityAnchor_Release_Data/data_whu/Whu_grid_0.005.tif")
            pc_file = os.path.join("./CityAnchor_Release_Data/data_whu/fused_whu_anno_v2.txt")

            object_id = i_2

            import rasterio
            with rasterio.open(raster_file) as src:
                image = src.read()
            image_with_rectangle = image[:, :, :]

            point_cloud = np.loadtxt(pc_file)

            bbox = bbox_data[str(object_id)].get("bbox")
            pixel = bbox_data[str(object_id)].get("pixel")

            name = info_data[str(object_id)].get("name")
            shp_name = info_data[str(object_id)].get("shp_name")

            x1 = pixel[0]
            y1 = pixel[1]
            x2 = pixel[2]
            y2 = pixel[3]

            X_pixel = int((x1 + x2) / 2)
            Y_pixel = int((y1 + y2) / 2)
            X_delta = int(x2 - x1)
            Y_delta = int(y2 - y1)

            try:
                heat_value = pred_mask_ROI[image.shape[1] - Y_pixel, X_pixel]
            except IndexError:
                heat_value = -100
            
            try:
                heat_value_nor = pred_mask_ROI_scaled[image.shape[1] - Y_pixel, X_pixel]
            except IndexError:
                heat_value_nor = -1
            
            if heat_value_nor < threshold_ROI:
                continue
            
            # prompt = query + " Please output segmentation mask."
            if name is not None:
                prompt_object_category = " The class of this object belongs to " + shp_name + "The name of this object is: " + name + ". Please describe this object highlighted by the red box. "
            if name is None:
                prompt_object_category = " The class of this object belongs to " + shp_name + ". Please describe this object highlighted by the red box. "

            prompt = prompt_object_category + query + " Please output segmentation mask."

            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            if args.use_mm_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()
    
            x1 = pixel[0]
            y1 = pixel[1]
            x2 = pixel[2]
            y2 = pixel[3]

            X_pixel = int((x1 + x2) / 2)
            Y_pixel = int((y1 + y2) / 2)
            X_delta = int(x2 - x1)
            Y_delta = int(y2 - y1)

            x1 = int(X_pixel - int(X_delta))
            y1 = int(image.shape[1] - Y_pixel - int(1.0 * Y_delta))

            x2 = int(X_pixel + int(X_delta))
            y2 = int(image.shape[1] - Y_pixel + int(1.0 * Y_delta))
            thickness = min(6, x2 - x1, y2 - y1)

            if min(x2 - x1, y2 - y1) < 100:
                Image_LH = 10 * max(X_delta, Y_delta)
                thickness = 8
        
            if min(x2 - x1, y2 - y1) > 100:
                Image_LH = 10 * max(X_delta, Y_delta)
                thickness = 20

            for t in range(thickness):
              try: 
                image_with_rectangle[0, y1:y2, x1 + t + 1] = 255
                image_with_rectangle[0, y1:y2, x2 - t - 1] = 255
                image_with_rectangle[0, y1 + t + 1, x1:x2] = 255
                image_with_rectangle[0, y2 - t - 1, x1:x2] = 255
                image_with_rectangle[1, y1:y2, x1 + t + 1] = 0
                image_with_rectangle[1, y1:y2, x2 - t - 1] = 0
                image_with_rectangle[1, y1 + t + 1, x1:x2] = 0
                image_with_rectangle[1, y2 - t - 1, x1:x2] = 0
                image_with_rectangle[2, y1:y2, x1 + t + 1] = 0
                image_with_rectangle[2, y1:y2, x2 - t - 1] = 0
                image_with_rectangle[2, y1 + t + 1, x1:x2] = 0
                image_with_rectangle[2, y2 - t - 1, x1:x2] = 0
              except IndexError:
                continue
          
            if min(x2 - x1, y2 - y1) >= 100:
                local_image = extract_subimage(image_with_rectangle, image.shape[1] - Y_pixel, X_pixel, 10 * Y_delta, 10 * X_delta)
            
            if min(x2 - x1, y2 - y1) < 100:
                local_image = extract_subimage(image_with_rectangle, image.shape[1] - Y_pixel, X_pixel, 20 * Y_delta, 20 * X_delta)

            local_image = np.transpose(local_image, (1, 2, 0))
            image_np = local_image
            original_size_list = [image_np.shape[:2]]
            try:
              image_clip = (
                clip_image_processor.preprocess(image_np, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
              )
            except ZeroDivisionError:
              continue
            
            if args.precision == "bf16":
                image_clip = image_clip.bfloat16()
            elif args.precision == "fp16":
                image_clip = image_clip.half()
            else:
                image_clip = image_clip.float()

            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]

            image = (
                preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
                .unsqueeze(0)
                .cuda()
            )
            if args.precision == "bf16":
                image = image.bfloat16()
            elif args.precision == "fp16":
                image = image.half()
            else:
                image = image.float()

            input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).cuda()

            pc_feats = []
            feat_instance = np.zeros((1, 1024))
            pc_feats.append(feat_instance)
            pc_feats = np.vstack(pc_feats)
            pc_feats = pc_feats[0]

            landmark_feats = []
            landmark_feat_instance  = np.zeros((1, 128))
            landmark_feats.append(landmark_feat_instance)
            landmark_feats = np.vstack(landmark_feats)
            landmark_feats = landmark_feats[0]

            # For nearest_pc
            try: 
              while len(nearest_instance_ids) < K: # K value 
                nearest_instance_ids.append(object_id)
            except AttributeError:
              nearest_instance_ids = []
              while len(nearest_instance_ids) < K: # K value 
                nearest_instance_ids.append(object_id)

            pc_feats_nearest = []
            for i in nearest_instance_ids:
                feat_instance = np.zeros((1, 1024))
                pc_feats_nearest.append(feat_instance)
            pc_feats_nearest = np.vstack(pc_feats_nearest)

            landmark_feats_nearest = []
            for i in nearest_instance_ids:
                landmark_feat_instance = np.zeros((1, 128))
                landmark_feats_nearest.append(landmark_feat_instance)
            landmark_feats_nearest = np.vstack(landmark_feats_nearest)
            
            lang_ans = model.evaluate(
            pc_feats,
            landmark_feats,
            pc_feats_nearest,
            landmark_feats_nearest,
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
            )

            lang_feat_list.append(lang_ans)
            object_list.append(i_2)

        list_combined = list(zip(object_list, lang_feat_list))
        list_sorted_combined = sorted(list_combined, key=lambda x: x[1], reverse=True)
        list_sorted_object, list_sorted_lang = zip(*list_sorted_combined)
        list_sorted_object = list(list_sorted_object)
        list_sorted_lang = list(list_sorted_lang)
        print("Sorted object:", list_sorted_object)
        print("Sorted lang:", list_sorted_lang)

        sorted_list = sorted(lang_feat_list)

        rank = 1
        for num in sorted_list:
            if lang_ans_object < num:
                rank += 1

        print("The rank is: ", rank)
        print("The object is: ", object_id_ori)
        
        with open(file_path, "a", encoding='utf-8') as file:
            file.write("object_id:" + str(object_id_ori) + ", ann_id:" + str(ann_id) + ", rank:" + str(rank) + ", pos:" + str(lang_ans_object) + "\n" )

    print("Test Completed")


def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes, train_scenes_to_use=None, val_scenes_to_use=None):
    # get initial scene list
    if train_scenes_to_use is not None:
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train if data["scene_id"] in train_scenes_to_use])))
    else:
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        
    if val_scenes_to_use is not None:
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val if data["scene_id"] in val_scenes_to_use])))
    else:        
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        
    if num_scenes == -1:
        num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= num_scenes
        
    # slice train_scene_list
    train_scene_list = train_scene_list[:num_scenes]

    # filter data in chosen scenes
    new_scanrefer_train = []
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            new_scanrefer_train.append(data)

    new_scanrefer_val = []
    for data in scanrefer_val:
        if data["scene_id"] in val_scene_list:
            new_scanrefer_val.append(data)

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list


## def for construction of our cityrefer dataset (Query, 3D instance, 2D image)
def one_hot(length, position):
    zeros = [0 for _ in range(length)]
    zeros[position] = 1
    zeros = np.array(zeros)
    return zeros


def shuffle_items_with_indices(lst):
    indices = list(range(len(lst)))  
    random.shuffle(indices) 
    shuffled_items = [lst[idx] for index, idx in enumerate(indices)]
    return shuffled_items, indices


def extract_subimage(image, center_x, center_y, width, height):

    start_x = max(center_x - width // 2, 0)
    end_x = min(center_x + width // 2, image.shape[1])
    start_y = max(center_y - height // 2, 0)
    end_y = min(center_y + height // 2, image.shape[2])
    subimage = image[:, int(start_x):int(end_x), int(start_y):int(end_y)]
    return subimage

def find_rows_with_value(arr, value):
    rows_with_value = np.where(arr == value)[0]
    return rows_with_value

def find_nearest_instances(arr, objectID,K):

    object_row = arr[arr[:, -1] == objectID]
    if object_row.shape[0] == 0:
        return []
    object_xyz = object_row[:, :3]
    distances = np.sqrt(np.sum((arr[:, :3] - object_xyz) ** 2, axis=1))
    distances[arr[:, -1] == objectID] = np.inf
    nearest_indices = np.argsort(distances)[:K]
    return arr[nearest_indices, -1]

def acc_cal_with_iou(test_file, iou_file_root):
    with open(test_file, 'r') as file:
        lines = file.readlines()

    total_count = 0
    rank_1_count = 0
    rank_le_2_count = 0
    rank_le_3_count = 0
    rank_le_5_count = 0
    rank_le_10_count = 0
    rank_1_acc25_count = 0
    rank_1_acc50_count = 0
    rank_2_acc25_count = 0
    rank_2_acc50_count = 0
    rank_3_acc25_count = 0
    rank_3_acc50_count = 0
    rank_5_acc50_count = 0
    rank_10_acc50_count = 0

    pattern = re.compile(r'(\d+_\w+_\w+), object_id:(\d+), ann_id:(\d+), rank:(\d+),')

    for line in lines:
        match = pattern.search(line)
        if match:
            scene_id, object_id, ann_id, rank = match.groups()
            iou_file = iou_file_root + scene_id + "_iou.json"
            with open(iou_file, 'r') as json_file:
                iou_scene = json.load(json_file)
            iou_object = iou_scene[str(object_id)]

            total_count += 1
            rank = int(rank)
            if rank == 1:
                rank_1_count += 1
                if iou_object >= 0.25:
                    rank_1_acc25_count += 1
                if iou_object >= 0.50:
                    rank_1_acc50_count += 1  
            if rank <= 2:
                rank_le_2_count += 1
                if iou_object >= 0.25:
                    rank_2_acc25_count += 1
                if iou_object >= 0.50:
                    rank_2_acc50_count += 1 
            if rank <= 3:
                rank_le_3_count += 1
                if iou_object >= 0.25:
                    rank_3_acc25_count += 1
                if iou_object >= 0.50:
                    rank_3_acc50_count += 1 
            if rank <= 5:
                rank_le_5_count += 1
                if iou_object >= 0.50:
                    rank_5_acc50_count += 1 
            if rank <= 10:
                rank_le_10_count += 1
                if iou_object >= 0.50:
                    rank_10_acc50_count += 1 

    total_count = len(lines)
    rank_1_ratio = rank_1_count / total_count if total_count > 0 else 0
    rank_le_2_ratio = rank_le_2_count / total_count if total_count > 0 else 0
    rank_le_3_ratio = rank_le_3_count / total_count if total_count > 0 else 0
    rank_le_5_ratio = rank_le_5_count / total_count if total_count > 0 else 0
    rank_le_10_ratio = rank_le_10_count / total_count if total_count > 0 else 0
    rank_1_acc25_ratio = rank_1_acc25_count / total_count if total_count > 0 else 0
    rank_1_acc50_ratio = rank_1_acc50_count / total_count if total_count > 0 else 0
    rank_2_acc25_ratio = rank_2_acc25_count / total_count if total_count > 0 else 0
    rank_2_acc50_ratio = rank_2_acc50_count / total_count if total_count > 0 else 0
    rank_3_acc25_ratio = rank_3_acc25_count / total_count if total_count > 0 else 0
    rank_3_acc50_ratio = rank_3_acc50_count / total_count if total_count > 0 else 0
    rank_5_acc50_ratio = rank_5_acc50_count / total_count if total_count > 0 else 0
    rank_10_acc50_ratio = rank_10_acc50_count / total_count if total_count > 0 else 0

    print(f"Sample of Acc@0.25(Rank 1): {rank_1_acc25_count}, Ratio of Acc@0.25(Rank 1): {rank_1_acc25_ratio:.2%}")
    print(f"Sample of Acc@0.50(Rank 1): {rank_1_acc50_count}, Ratio of Acc@0.50(Rank 1): {rank_1_acc50_ratio:.2%}")
    print(f"Sample of Acc@0.50(Rank 2): {rank_2_acc50_count}, Ratio of Acc@0.50(Rank 2): {rank_2_acc50_ratio:.2%}")
    print(f"Sample of Acc@0.50(Rank 3): {rank_3_acc50_count}, Ratio of Acc@0.50(Rank 3): {rank_3_acc50_ratio:.2%}")
    print(f"Sample of Acc@0.50(Rank 5): {rank_3_acc50_count}, Ratio of Acc@0.50(Rank 5): {rank_5_acc50_ratio:.2%}")
    print(f"Sample of Acc@0.50(Rank 10): {rank_3_acc50_count}, Ratio of Acc@0.50(Rank 10): {rank_10_acc50_ratio:.2%}")

    return 1

if __name__ == "__main__":
    main(sys.argv[1:])
