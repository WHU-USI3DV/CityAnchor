import glob
import json
import os
import random
import re

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST)

from lib.config import CONF
from data.sensaturban.model_util_sensaturban import SensatUrbanDatasetConfig

from PIL import Image
import rasterio
from rasterio.plot import show
from utils.pc_utils import random_sampling, rotx, roty, rotz

class ReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        reason_seg_data, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        images = []
        jsons = []
        if reason_seg_data == 'ReasonSeg':
            for split in splits:
                images_split = glob.glob(
                    os.path.join(
                        base_image_dir, "reason_seg", reason_seg_data, split, "*.jpg"
                    )
                )
                images.extend(images_split)
            jsons = [path.replace(".jpg", ".json") for path in images]
        elif reason_seg_data == 'rsvg':
            for split in splits:
                jsons = glob.glob(
                    os.path.join(
                        base_image_dir, 'RSVG', reason_seg_data, 'Json', split, '*.json'
                    )
                )
            imagePath = os.path.join(
                base_image_dir,'RSVG', reason_seg_data, 'images'
            )
            for path in jsons:
                name = os.path.basename(path)
                name = name.replace('.json', '.jpg')
                if '(' in name:
                    name = re.sub('\([0-9]\)', '', name)
                images.append(os.path.join(imagePath, name))
        elif reason_seg_data == 'SensatUrban-LISA-EX':
          for split in splits:
            images_split = glob.glob(
              os.path.join(
                base_image_dir,'SensatUrban-LISA-EX',split,'*.jpg'
              )
            ) 
            images.extend(images_split)
          jsons = [path.replace(".jpg", ".json") for path in images]
        
        elif reason_seg_data == 'UrbanBIS':
          for split in splits:
            json_split = glob.glob(
              os.path.join(
                base_image_dir,'UrbanBIS',split,'*.json'
              )
            )
            jsons.extend(json_split)
          images = [ 
            x.split('_')[0] + "_" + x.split('_')[1]+'.tif' 
            for x in jsons
          ]
          print(images[0])
          

        self.reason_seg_data = (images, jsons)
        print("number of reason_seg samples: ", len(images))

        if explanatory != -1 and reason_seg_data == 'ReasonSeg':
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST
            self.img_to_explanation = {}
            with open(
                os.path.join(
                    base_image_dir,
                    "reason_seg",
                    reason_seg_data,
                    "explanatory",
                    "train.json",
                )
            ) as f:
                items = json.load(f)
            for item in items:
                img_name = item["image"]
                self.img_to_explanation[img_name] = {
                    "query": item["query"],
                    "outputs": item["outputs"],
                }

            print("len(self.img_to_explanation): ", len(self.img_to_explanation))

        print("preparing data...")
        DC = SensatUrbanDatasetConfig() 
        SCANREFER_TRAIN = json.load(open(os.path.join("./CityAnchor_Release_Data/meta_data/CityRefer_train_all_v1.json"))) 
        SCANREFER_VAL = json.load(open(os.path.join("./CityAnchor_Release_Data/meta_data/CityRefer_val_NO.json"))) 
        scanrefer_train, scanrefer_val, all_scene_list= get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, -1)
        scanrefer = {"train": scanrefer_train, "val": scanrefer_val}
        self.scanrefer=scanrefer["train"] # train or val
        self.DC=DC
        self.other_object_cat = -1
        self._load_sensaturban_data()     
        self.CONF = CONF
        self.use_cache = False
        self.cache = {}
        self.num_cands = 3
        self.scanrefer_all_scene = all_scene_list
        self.num_points = -1
        self.use_color = True
        self.use_height = True
        self.augment = False
        self.use_landmark = True
        self.num_inst_points = 4096
        self.other_object_cat = -1
        self.voxel_size_ap = 0.33
        self.voxel_size_glp = 0.33
        self.no_gt_instance = 192
        self.max_num_object = 600
        self.max_num_landmark = 192
        self.split = "train"
        self.no_gt_instance = False
        self.K = 5 # K change this

    def _get_raw2label(self, SCANNET_V2_TSV):
        # Mapping
        scannet_labels = self.DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    # def for cityrefer dataset
    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = int(data["object_id"])
            object_name = " ".join(data["object_name"].split("_"))

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(self.other_object_cat)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = int(data["object_id"])
            object_name = data["object_name"] # Ground, Building, Parking, Car...
            ann_id = int(data["ann_id"])

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = self.other_object_cat

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _load_sensaturban_data(self):
        print("Loading data...")
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))
        self.raw2label = {self.DC.type2class[class_ind]:class_ind for class_ind in self.DC.type2class.keys()}
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]
        # translation factors
        factor = (torch.rand(3) - 0.5).tolist()
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox


    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ### For original llava 
        images, jsons = self.reason_seg_data
        idx = random.randint(0, len(images) - 1)

        ### Getting item for CityRefer dataset
        idx_2 = random.randint(0, len(self.scanrefer) - 1)
        scene_id = self.scanrefer[idx_2]["scene_id"]
        object_id = int(self.scanrefer[idx_2]["object_id"])
        object_name = self.scanrefer[idx_2]["object_name"]
        ann_id = int(self.scanrefer[idx_2]["ann_id"])
        object_cat = self.raw2label[object_name] if object_name in self.raw2label else self.other_object_cat
        assert object_cat >= 0
        assert object_id >= 0
        MAX_NUM_OBJ = self.max_num_object
        MAX_NUM_LANDMARK = self.max_num_landmark
        query = self.scanrefer[idx_2]["description"]

        local_image = np.zeros((3, 9, 9))
        if self.use_cache and scene_id in self.cache:
            mesh_vertices, instance_labels, semantic_labels, instance_bboxes, landmark_names, landmark_ids, globalShift = self.cache[scene_id]
        else:
            pg_file = os.path.join(self.CONF.PATH.SCAN_DATA, scene_id+".pth")
            feat_file = os.path.join(self.CONF.PATH.SCAN_DATA, scene_id+".json")
            landmark_file = os.path.join(self.CONF.PATH.SCAN_DATA, scene_id+"_landmark.json")
            landmark_nearest_file = os.path.join(self.CONF.PATH.LANDMARK_DATA, scene_id+".landmark_feat.json")

            coords, colors, label_ids, instance_ids, label_ids_pg, instance_ids_pg, instance_bboxes, \
                landmark_names, landmark_ids, globalShift = torch.load(pg_file)
            with open(feat_file, 'r') as json_file:
                feats = json.load(json_file)
            
            with open(landmark_file, 'r') as json_file:
                landmark_feat = json.load(json_file)

            with open(landmark_nearest_file, 'r') as json_file:
                landmark_nearest_feat = json.load(json_file) 

            mesh_vertices = np.concatenate([coords, colors], axis=1) 
            instance_bboxes = np.stack([instance_bboxes[instance_id] for instance_id in sorted(instance_bboxes.keys()) if instance_id != -100])
            if self.no_gt_instance:
                instance_labels = instance_ids_pg
                semantic_labels = label_ids_pg
            else:                
                instance_labels = instance_ids
                semantic_labels = label_ids
            if self.use_cache:
                self.cache[scene_id] = mesh_vertices, instance_labels, semantic_labels, instance_bboxes, \
                    landmark_names, landmark_ids, globalShift

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:6] = point_cloud[:, 3:6] / 127.5 - 1 

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)
            
        if self.num_points > 0:
            point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
            instance_labels = instance_labels[choices]
            semantic_labels = semantic_labels[choices]

        instance_class = []
        pc_feats = []

        cand_instance_ids = [cand_id for cand_id in np.unique(instance_labels) if cand_id != -100]
        nearest_instance_ids = find_nearest_instances(instance_bboxes, object_id, self.K) # Finding nearest instances
        if self.num_cands > 0:
            if object_id in cand_instance_ids:
                cand_instance_ids.remove(object_id)    
                cand_instance_ids = sorted([object_id] + random.sample(cand_instance_ids, min(self.num_cands - 1, len(cand_instance_ids))))
            else:
                cand_instance_ids = sorted(random.sample(cand_instance_ids, min(self.num_cands, len(cand_instance_ids))))
        
        cand_instance_ids = np.random.permutation(cand_instance_ids)
        item_id = cand_instance_ids[0] # Randomly select positive and negative samples

        ### Multi-modality embedding for single instance
        # Image - Top view feat
        raster_file = os.path.join(self.CONF.PATH.SCAN_DATA, scene_id+".tif")
        with rasterio.open(raster_file) as src:
            image = src.read()
      
        instance_bboxes_pixel = np.copy(instance_bboxes)
        instance_bboxes_pixel[:,0] = instance_bboxes_pixel[:,0] / 0.1
        instance_bboxes_pixel[:,1] = instance_bboxes_pixel[:,1] / 0.1
        instance_bboxes_pixel[:,3] = instance_bboxes_pixel[:,3] / 0.1
        instance_bboxes_pixel[:,4] = instance_bboxes_pixel[:,4] / 0.1
        arr = instance_bboxes[:,-1]
        row = find_rows_with_value(arr, item_id)
        if len(row) == 0:
            subimage = np.zeros((256,256,3))
        else:         
            X_pixel = int(instance_bboxes_pixel[row,0])
            Y_pixel = int(instance_bboxes_pixel[row,1])
            X_delta = int(instance_bboxes_pixel[row,3])
            Y_delta = int(instance_bboxes_pixel[row,4])
            Image_LH = max(X_delta,Y_delta)
            local_image = extract_subimage(image,  image.shape[2] - Y_pixel, X_pixel, Image_LH, Image_LH)

        local_image = np.transpose(local_image, (1, 2, 0))

        # PC - Uni3d feat
        pc_feats = []
        landmark_feats = []
        ind = np.nonzero(instance_labels == item_id)[0]
        ins_class = semantic_labels[ind[0]]
        x = point_cloud[ind]
        ins_class = self.DC.label_id2class[int(ins_class)]
        instance_class.append(ins_class)
        pc = x[:, :3]
        center = 0.5 * (pc.min(0) + pc.max(0))
        size = pc.max(0) - pc.min(0)
        ins_obb = np.concatenate((center, size, np.array([0])))
        if self.num_inst_points > 0:
            x = random_sampling(x, self.num_inst_points)
        pc = x[:, :3]
          
        try:
            feat_instance = feats[str(int(item_id))]
        except KeyError:
            feat_instance = np.zeros((1, 1024)) # Unable to equip pre-trained pc_feats for object

        pc_feats.append(feat_instance)
        pc_feats = np.vstack(pc_feats)

        # Landmark - BiGRU feat
        try:
            landmark_feat_instance = landmark_feat[str(int(item_id))]
        except KeyError:
            landmark_feat_instance = np.zeros((1, 128)) # Unable to equip pre-trained pc_feats for object
        
        landmark_feats.append(landmark_feat_instance)
        landmark_feats = np.vstack(landmark_feats)

        # K-nearest objects
        while len(nearest_instance_ids) < self.K: # K value 
            nearest_instance_ids.append(object_id)

        # K-nearest objects - uni3d feat
        pc_feats_nearest = []
        for i in nearest_instance_ids:
            feat_instance = np.zeros((1, 1024))
            try:
                feat_instance = feats[str(int(i))]
            except KeyError:
                feat_instance = np.zeros((1, 1024))
            pc_feats_nearest.append(feat_instance)

        pc_feats_nearest = np.vstack(pc_feats_nearest)

        # K-nearest landmark - BiGRU feat
        landmark_feats_nearest = []
        for i in nearest_instance_ids:
            landmark_feat_instance = np.zeros((1, 128))
            try:
                landmark_feat_instance = landmark_nearest_feat[str(int(i))]
            except KeyError:
                landmark_feat_instance = np.zeros((1, 128))
            landmark_feats_nearest.append(landmark_feat_instance)

        landmark_feats_nearest = np.vstack(landmark_feats_nearest)

        # GT
        lang_ans =  []
        for i_instance in cand_instance_ids:
            if (i_instance == object_id) and (i_instance == item_id):
                lang_ans.append(1)
            else:
                lang_ans.append(0)

        lang_gt = lang_ans[0]
        pc_feats = pc_feats[0]
        landmark_feats = landmark_feats[0]

        # Using LLaVA as backbone for CityAnchor
        image_path = images[idx] 
        json_path = jsons[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image = local_image
        ori_size = image.shape[:2]
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        mask, sents, is_sentence = get_mask_from_json(json_path, image)
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ]

        image = self.transform.apply_image(image)  # preprocess image for sam (No need for stage 2)
        resize = image.shape[:2]

        image_name = image_path.split("/")[-1]
        if self.explanatory != -1 and image_name in self.img_to_explanation:
            if random.random() < self.explanatory:
                choice = 2
            else:
                choice = random.randint(0, 1)
        
        # Language embedding
        questions = []
        answers = []
        query_list = []
        query_list.append(query)
        
        for text in query_list:
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))
            else:
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

            # add explanation if applicable 
            img_name = image_path.split("/")[-1]
            if self.explanatory != -1 and img_name in self.img_to_explanation:
                if choice == 0:  # [SEG] token
                    #  answers.append(random.choice(self.answer_list))
                    answers.append(self.answer_list[0])
                elif choice == 1:  # [SEG] token + text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    answer = random.choice(self.answer_list) + " {}".format(answer)
                    questions[-1] = (
                        DEFAULT_IMAGE_TOKEN
                        + "\n"
                        + text
                        + " {}".format(random.choice(self.explanatory_question_list))
                    )
                    # answers.append(answer)
                    answers.append(self.answer_list[0])
                elif choice == 2:  # vanilla text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text
                    # answers.append(answer)
                    answers.append(self.answer_list[0])
                else:
                    raise ValueError("Not implemented yet.")
            else:
                # answers.append(random.choice(self.answer_list))
                answers.append(self.answer_list[0])

            conversations = []
            conv = conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        image_name = image_path.split("/")[-1]
        if (
            self.explanatory != -1
            and image_name in self.img_to_explanation
            and choice == 2
        ):
            masks = torch.rand(0, *ori_size)
            label = torch.ones(ori_size) * self.ignore_label
        else:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_sents, 
            lang_gt,  
            pc_feats, 
            landmark_feats,
            pc_feats_nearest,
            landmark_feats_nearest,
        )

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

    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list


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
    end_x = min(center_x + width // 2, image.shape[2])
    start_y = max(center_y - height // 2, 0)
    end_y = min(center_y + height // 2, image.shape[1])
    subimage = image[:,  start_x:end_x,start_y:end_y]
    
    return subimage

def find_rows_with_value(arr, value):
    rows_with_value = np.where(arr == value)[0]
    return rows_with_value

def find_nearest_instances(arr, objectID, K):
    object_row = arr[arr[:, -1] == objectID]
    if object_row.shape[0] == 0:
        return []
    object_xyz = object_row[:, :3]
    distances = np.sqrt(np.sum((arr[:, :3] - object_xyz) ** 2, axis=1))
    distances[arr[:, -1] == objectID] = np.inf
    nearest_indices = np.argsort(distances)[:K]
    return arr[nearest_indices, -1]

