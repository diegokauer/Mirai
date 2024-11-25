import functools
import logging
import os
import pickle
import tempfile
import traceback
from typing import List, BinaryIO
import warnings
import zipfile

import numpy as np
import pydicom
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)

import onconet.transformers.factory as transformer_factory
import onconet.models.calibrator
import onconet.utils.dicom
from onconet import __version__ as onconet_version
from onconet.models.factory import load_model, RegisterModel, get_model_by_name
from onconet.models.factory import get_model
from onconet.transformers.basic import ComposeTrans
from onconet.utils import parsing
from onconet.utils.logging_utils import get_logger


@RegisterModel("mirai_full")
class MiraiFull(nn.Module):

    def __init__(self, args):
        super(MiraiFull, self).__init__()
        self.args = args
        if args.img_encoder_snapshot is not None:
            self.image_encoder = load_model(args.img_encoder_snapshot, args, do_wrap_model=False)
        else:
            self.image_encoder = get_model_by_name('custom_resnet', False, args)

        if hasattr(self.args, "freeze_image_encoder") and self.args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.image_repr_dim = self.image_encoder._model.args.img_only_dim
        if args.transformer_snapshot is not None:
            self.transformer = load_model(args.transformer_snapshot, args, do_wrap_model=False)
        else:
            args.precomputed_hidden_dim = self.image_repr_dim
            self.transformer = get_model_by_name('transformer', False, args)
        args.img_only_dim = self.transformer.args.transfomer_hidden_dim

    def forward(self, x, risk_factors=None, batch=None):
        B, C, N, H, W = x.size()
        x = x.transpose(1,2).contiguous().view(B*N, C, H, W)
        risk_factors_per_img =  (lambda N, risk_factors: [factor.expand( [N, *factor.size()]).contiguous().view([-1, factor.size()[-1]]).contiguous() for factor in risk_factors])(N, risk_factors) if risk_factors is not None else None
        _, img_x, _ = self.image_encoder(x, risk_factors_per_img, batch)
        img_x = img_x.view(B, N, -1)
        img_x = img_x[:,:,: self.image_repr_dim]
        logit, transformer_hidden, activ_dict = self.transformer(img_x, risk_factors, batch)
        return logit, transformer_hidden, activ_dict


def download_file(url, destination):
    import urllib.request

    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as e:
        get_logger().error(f"An error occurred while downloading from {url} to {destination}: {e}")
        raise e


def _torch_set_num_threads(threads) -> int:
    """
    Set the number of CPU threads for torch to use.
    Set to a negative number for no-op.
    Set to 0 for the number of CPUs.
    """
    if threads < 0:
        return torch.get_num_threads()
    if threads is None or threads == 0:
        # I've never seen a benefit to going higher than 8 and sometimes there is a big slowdown
        threads = min(8, os.cpu_count())

    torch.set_num_threads(threads)
    return torch.get_num_threads()


class MiraiModel:
    """
    Represents a trained Mirai model. Useful for predictions on individual exams.
    """
    def __init__(self, config_obj):
        super().__init__()
        self.args = self.sanitize_paths(config_obj)
        self.__version__ = onconet_version
        self._model = None
        self._calibrator = None
        self._device = None

    def to(self, device):
        if self._model:
            self._model.to(device)
            self._model.transformer.to(device)
        self._device = device
        return self

    def get_device(self):
        if self._device:
            return self._device
        return get_default_device()

    def load_model(self):
        if self._model:
            return self._model

        logger = get_logger()
        logger.debug("Loading model...")
        self.args.cuda = self.args.cuda and torch.cuda.is_available()

        self.download_if_needed(self.args)
        if self.args.model_name == 'mirai_full':
            model = get_model(self.args)
        else:
            model = torch.load(self.args.snapshot, map_location='cpu')

        # Unpack models that were trained as data parallel
        if isinstance(model, nn.DataParallel):
            model = model.module

        # Add use precomputed hiddens for models trained before it was introduced.
        # Assumes a resnet WHybase backbone
        try:
            model._model.args.use_precomputed_hiddens = self.args.use_precomputed_hiddens
            model._model.args.cuda = self.args.cuda
        except Exception as e:
            logger.debug("Exception caught, skipping precomputed hiddens")
            pass

        self._model = model
        return model

    def load_calibrator(self):
        if self._calibrator:
            return self._calibrator

        get_logger().debug("Loading calibrator...")

        # Load calibrator if desired
        if self.args.calibrator_path is not None:
            with open(self.args.calibrator_path, 'rb') as infi:
                calibrator = pickle.load(infi)
        else:
            calibrator = None

        self._calibrator = calibrator
        return calibrator

    def process_image_joint(self, batch, model, calibrator, risk_factor_vector=None):
        logger = get_logger()
        logger.debug("Getting predictions...")

        if self.args.cuda:
            device = self.get_device()
            logger.debug(f"Inference with {device}")
            self.to(device)
            for key, val in batch.items():
                batch[key] = val.to(device)
        else:
            model = model.cpu()
            logger.debug("Inference with CPU")

        risk_factors = autograd.Variable(risk_factor_vector.unsqueeze(0)) if risk_factor_vector is not None else None

        logit, _, _ = model(batch['x'], risk_factors, batch)
        probs = F.sigmoid(logit).cpu().data.numpy()
        pred_y = np.zeros(probs.shape[1])

        if calibrator is not None:
            logger.debug("Raw probs: {}".format(probs))

            for i in calibrator.keys():
                pred_y[i] = calibrator[i].predict_proba(probs[0, i].reshape(-1, 1)).flatten()[1]

        return pred_y.tolist()

    def process_exam(self, images, risk_factor_vector):
        if len(images) != 4:
            raise ValueError(f"Require exactly 4 images, instead we got {len(images)}")

        logger = get_logger()
        logger.debug(f"Processing images...")

        test_image_transformers = parsing.parse_transformers(self.args.test_image_transformers)
        test_tensor_transformers = parsing.parse_transformers(self.args.test_tensor_transformers)
        test_transformers = transformer_factory.get_transformers(test_image_transformers, test_tensor_transformers, self.args)
        transforms = ComposeTrans(test_transformers)

        batch = self.collate_batch(images, transforms)
        model = self.load_model()
        calibrator = self.load_calibrator()

        y = self.process_image_joint(batch, model, calibrator, risk_factor_vector)

        return y

    def collate_batch(self, images, transforms):
        get_logger().debug("Collating batches...")

        batch = {}
        batch['side_seq'] = torch.cat([torch.tensor(b['side_seq']).unsqueeze(0) for b in images], dim=0).unsqueeze(0)
        batch['view_seq'] = torch.cat([torch.tensor(b['view_seq']).unsqueeze(0) for b in images], dim=0).unsqueeze(0)
        batch['time_seq'] = torch.zeros_like(batch['view_seq'])

        batch['x'] = torch.cat(
            (lambda imgs: [transforms(b['x']).unsqueeze(0) for b in imgs])(images), dim=0
        ).unsqueeze(0).transpose(1, 2)

        return batch

    def run_model(self, dicom_files: List[BinaryIO], payload=None, is_dicom=True):
        logger = get_logger()
        _torch_set_num_threads(getattr(self.args, 'threads', 0))
        if payload is None:
            payload = dict()

        dcmread_force = payload.get("dcmread_force", False)
        dcmtk_installed = onconet.utils.dicom.is_dcmtk_installed()
        use_dcmtk = payload.get("dcmtk", False) and dcmtk_installed
        if use_dcmtk:
            logger.debug('Using dcmtk')
        else:
            logger.debug('Using pydicom')

        images = []
        dicom_info = {}
        if is_dicom:
            for dicom in dicom_files:
                try:
                    cur_dicom = pydicom.dcmread(dicom, force=dcmread_force, stop_before_pixels=True)
                    view, side = onconet.utils.dicom.get_dicom_info(cur_dicom)

                    if (view, side) in dicom_info:
                        prev_dicom = pydicom.dcmread(dicom_info[(view, side)], force=dcmread_force, stop_before_pixels=True)
                        prev = int(prev_dicom[0x0008, 0x0023].value + prev_dicom[0x0008, 0x0033].value)
                        cur = int(cur_dicom[0x0008, 0x0023].value + cur_dicom[0x0008, 0x0033].value)

                        if cur > prev:
                            dicom_info[(view, side)] = dicom
                    else:
                        dicom_info[(view, side)] = dicom
                except Exception as e:
                    logger.warning(f"Error reading DICOM: {e}")
                    logger.warning(f"{traceback.format_exc()}")

            for k in dicom_info:
                try:
                    dicom = dicom_info[k]
                    dicom.seek(0)
                    view, side = k

                    if use_dcmtk:
                        dicom_file = tempfile.NamedTemporaryFile(suffix='.dcm')
                        image_file = tempfile.NamedTemporaryFile(suffix='.png')
                        dicom_path = dicom_file.name
                        image_path = image_file.name
                        logger.debug("Temp DICOM path: {}".format(dicom_path))
                        logger.debug("Temp image path: {}".format(image_path))

                        dicom_file.write(dicom.read())

                        image = onconet.utils.dicom.dicom_to_image_dcmtk(dicom_path, image_path)
                        logger.debug('Image mode from dcmtk: {}'.format(image.mode))
                        images.append({'x': image, 'side_seq': side, 'view_seq': view})
                    else:
                        dicom = pydicom.dcmread(dicom, force=dcmread_force)
                        window_method = payload.get("window_method", "minmax")
                        image = onconet.utils.dicom.dicom_to_arr(dicom, window_method=window_method, pillow=True)
                        logger.debug('Image mode from dicom: {}'.format(image.mode))
                        images.append({'x': image, 'side_seq': side, 'view_seq': view})
                except Exception as e:
                    logger.warning(f"{type(e).__name__}: {e}")
                    logger.warning(f"{traceback.format_exc()}")
        elif not is_dicom:
            for png in dicom_files:
                view, side = png.replace('.png', '').split('_')[2:]
                dicom_info[(view, side)] = png

            for k in dicom_info:
                try:
                    png = dicom_info[k]
                    png.seek(0)
                    view, side = k
                    image = onconet.utils.dicom.png_to_arr(png)
                    logger.debug('Image mode from PNG: {}'.format(image.mode))
                    images.append({'x': image, 'side_seq': side, 'view_seq': view})
                except Exception as e:
                    logger.warning(f"{type(e).__name__}: {e}")
                    logger.warning(f"{traceback.format_exc()}")

        risk_factor_vector = None

        y = self.process_exam(images, risk_factor_vector)
        logger.debug(f'Raw Predictions: {y}')

        y = {'Year {}'.format(i+1): round(p, 4) for i, p in enumerate(y)}
        report = {'predictions': y}

        return report

    @staticmethod
    def sanitize_paths(args):
        path_keys = ["img_encoder_snapshot", "transformer_snapshot", "calibrator_path"]
        for key in path_keys:
            if hasattr(args, key) and getattr(args, key) is not None:
                setattr(args, key, os.path.expanduser(getattr(args, key)))
        return args

    @staticmethod
    def download_if_needed(args, cache_dir='./.cache'):
        args = MiraiModel.sanitize_paths(args)
        if args.model_name == 'mirai_full':
            if os.path.exists(args.img_encoder_snapshot) and os.path.exists(args.transformer_snapshot):
                return
        else:
            if os.path.exists(args.snapshot):
                return

        if getattr(args, 'remote_snapshot_uri', None) is None:
            return

        logger = get_logger()
        logger.info(f"Local models not found, downloading snapshot from remote URI: {args.remote_snapshot_uri}")
        os.makedirs(cache_dir, exist_ok=True)
        tmp_zip_path = os.path.join(cache_dir, "snapshots.zip")
        if not os.path.exists(tmp_zip_path):
            logger.debug(f"Downloading snapshot to {tmp_zip_path}")
            download_file(args.remote_snapshot_uri, tmp_zip_path)
        else:
            logger.debug(f"Snapshot already downloaded to {tmp_zip_path}")

        dest_dir = os.path.dirname(args.img_encoder_snapshot) if args.model_name == 'mirai_full' else os.path.dirname(args.snapshot)
        os.makedirs(dest_dir, exist_ok=True)

        # Unzip file
        logger.debug(f"Saving models to {dest_dir}")
        with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

        os.remove(tmp_zip_path)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        # Not all operations implemented in MPS yet
        use_mps = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"
        if use_mps:
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')
