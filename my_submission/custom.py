import numpy as np

import yaml

import numpy as np
import torch
import torchvision.models as models

from collections import OrderedDict

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from tqdm import tqdm

import sys
sys.path.append('./MCS2023_baseline/')
sys.path.append('./CGD')

from CGD.model import Model

from data_utils.augmentations import get_val_aug
from data_utils.dataset import SubmissionDataset
from utils import convert_dict_to_tuple

class Ranker:
    def __init__(self, dataset_path, gallery_csv_path, queries_csv_path):
        """
        Initialize your model here
        Inputs:
            dataset_path
            gallery_csv_path
            queries_csv_path
        """
        # Try not to change
        self.dataset_path = dataset_path
        self.gallery_csv_path = gallery_csv_path
        self.queries_csv_path = queries_csv_path
        self.max_predictions = 1000

        # Add your code below

        checkpoint_path = './MCS2023_baseline/experiments/baseline_mcs/baseline_model.pth'
        self.batch_size = 256

        self.exp_cfg = './MCS2023_baseline/config/baseline_mcs.yml'
        self.inference_cfg = './MCS2023_baseline/config/inference_config.yml'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(self.exp_cfg) as f:
            data = yaml.safe_load(f)
        self.exp_cfg = convert_dict_to_tuple(data)

        with open(self.inference_cfg) as f:
            data = yaml.safe_load(f)
        self.inference_cfg = convert_dict_to_tuple(data)

        print('Creating model and loading checkpoint')
        
        # num classes
        # products-10k: 9691
        # sop: 3...

        config = dict({
            'backbone_type': 'resnext50',
            'gd_config': 'GS',
            'feature_dim': 512,
            'num_classes': 9691,
        })

        # config = dict({
        #     'backbone_type': 'efficientnet-b4',
        #     'gd_config': 'GS',
        #     'feature_dim': 512,
        #     'num_classes': 9691,
        # })

        self.model = Model(**config)
        # ckpt = '/home/worker/workspace/hacks/mcs_2023/CGD/results/isc_uncropped_resnext50_GS_512_0.1_0.5_0.1_256_model.pth'
        # self.model = models.resnet50(pretrained=False, num_classes=5270)
        # ckpt = 'my_submission/weights.pt'
        # ckpt = 'my_submission/weights2.pt'
        ckpt = '../CGD/results/products-10k_uncropped_resnext50_SG_512_0.1_0.5_0.1_256_model_best.pth'
        # {'mAP': 0.18936997723867746}
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)
        self.embedding_shape = config['feature_dim']
        # self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model.to(self.device)
        print('Weights are loaded, fc layer is deleted!')


    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def predict_product_ranks(self):
        """
        This function should return a numpy array of shape `(num_queries, 1000)`. 
        For ach query image your model will need to predict 
        a set of 1000 unique gallery indexes, in order of best match first.

        Outputs:
            class_ranks - A 2D numpy array where the axes correspond to:
                          axis 0 - Batch size
                          axis 1 - An ordered rank list of matched image indexes, most confident prediction first
                            - maximum length of this should be 1000
                            - predictions above this limit will be dropped
                            - duplicates will be dropped such that the lowest index entry is preserved
        """

        gallery_dataset = SubmissionDataset(
            root=self.dataset_path, annotation_file=self.gallery_csv_path,
            transforms=get_val_aug(self.exp_cfg)
        )

        gallery_loader = torch.utils.data.DataLoader(
            gallery_dataset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True, num_workers=self.inference_cfg.num_workers
        )

        query_dataset = SubmissionDataset(
            root=self.dataset_path, annotation_file=self.queries_csv_path,
            transforms=get_val_aug(self.exp_cfg), with_bbox=True
        )

        query_loader = torch.utils.data.DataLoader(
            query_dataset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True, num_workers=self.inference_cfg.num_workers
        )

        print('Calculating embeddings')
        gallery_embeddings = np.zeros((len(gallery_dataset), self.embedding_shape))
        query_embeddings = np.zeros((len(query_dataset), self.embedding_shape))

        with torch.no_grad():
            for i, images in tqdm(enumerate(gallery_loader),
                                total=len(gallery_loader)):
                images = images.to(self.device)
                # outputs = self.model(images)
                outputs = self.model(images)[0]
                # outputs = outputs.data.cpu().numpy()
                outputs = outputs.cpu().numpy()

                gallery_embeddings[
                    i*self.batch_size:(i*self.batch_size + self.batch_size), :
                ] = outputs
            # baseline: 0.016
            # new model: 0.058
            for i, images in tqdm(enumerate(query_loader),
                                total=len(query_loader)):
                images = images.to(self.device)
                # outputs = self.model(images)
                outputs = self.model(images)[0]
                # outputs = outputs.data.cpu().numpy()
                outputs = outputs.cpu().numpy()
                query_embeddings[
                    i*self.batch_size:(i*self.batch_size + self.batch_size), :
                ] = outputs
        
        print('Normalizing and calculating distances')
        gallery_embeddings = normalize(gallery_embeddings)
        query_embeddings = normalize(query_embeddings)
        distances = pairwise_distances(query_embeddings, gallery_embeddings)
        sorted_distances = np.argsort(distances, axis=1)[:, :1000]

        class_ranks = sorted_distances
        return class_ranks