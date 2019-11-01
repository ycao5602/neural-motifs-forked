"""
File that involves dataloaders for the Visual Genome dataset.
"""

import json
import os

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from dataloaders.blob import Blob
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from config import VG_IMAGES, IM_DATA_FN, VG_SGG_FN, VG_SGG_DICT_FN, BOX_SCALE, \
    IM_SCALE, PROPOSAL_FN, GRAPH_DIR, VOCAB_DIR, DATA_PATH
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
    RandomOrder, Hue, random_crop
from collections import defaultdict
from pycocotools.coco import COCO
import xml.etree.ElementTree as ET
import scipy.sparse
from tqdm import tqdm



class VG(Dataset):
    def __init__(self, mode, roidb_file=VG_SGG_FN, dict_file=VG_SGG_DICT_FN,
                 image_file=IM_DATA_FN, filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True, #num_im=-1
                 use_proposals=False):
        """
        Torch dataset for VisualGenome
        :param mode: Must be train, test, or val
        :param roidb_file:  HDF5 containing the GT boxes, classes, and relationships
        :param dict_file: JSON Contains mapping of classes/relationships to words
        :param image_file: HDF5 containing image filenames
        :param filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
        :param filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
        :param num_im: Number of images in the entire dataset. -1 for all images.
        :param num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        :param proposal_file: If None, we don't provide proposals. Otherwise file for where we get RPN
            proposals
        """
        if mode not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))
        self.mode = mode
        # self.mode = 'val'
        print('mode',self.mode)

        # Initialize
        self.roidb_file = roidb_file
        self.dict_file = dict_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap
        self.filter_duplicate_rels = filter_duplicate_rels and self.mode == 'train'

        # self.split_mask = load_split(
        #     self.roidb_file, self.mode, num_im, num_val_im=num_val_im,
        #     filter_empty_rels=filter_empty_rels,
        # )


        # self.ind_to_classes, self.ind_to_predicates = load_info(dict_file)
        self.ind_to_classes = {}
        self.ind_to_predicates= {}

        # Load classes
        self._classes = ['__background__']
        self._class_to_ind = {}
        self._class_to_ind[self._classes[0]] = 0
        self.ind_to_classes = {}
        self.ind_to_classes[self._classes[0]] = 0
        with open(os.path.join(VOCAB_DIR, 'objects_vocab.txt')) as f:
            count = 1
            for object in f.readlines():
                names = [n.lower().strip() for n in object.split(',')]
                self._classes.append(names[0])
                for n in names:
                    self._class_to_ind[n] = count
                self.ind_to_classes[names[0]] = count
                count += 1

        # print('num_classes',len(self.ind_to_classes))
        # Load attributes
        self._attributes = ['__no_attribute__']
        self._attribute_to_ind = {}
        self._attribute_to_ind[self._attributes[0]] = 0
        with open(os.path.join(VOCAB_DIR, 'attributes_vocab.txt')) as f:
            count = 1
            for att in f.readlines():
                names = [n.lower().strip() for n in att.split(',')]
                self._attributes.append(names[0])
                for n in names:
                    self._attribute_to_ind[n] = count
                count += 1

        # Load relations
        self._relations = ['__no_relation__']
        self._relation_to_ind = {}
        self._relation_to_ind[self._relations[0]] = 0
        self.ind_to_predicates= {}
        self.ind_to_predicates[self._relations[0]]=0
        with open(os.path.join(VOCAB_DIR, 'relations_vocab.txt')) as f:
            count = 1
            for rel in f.readlines():
                names = [n.lower().strip() for n in rel.split(',')]
                self._relations.append(names[0])
                for n in names:
                    self._relation_to_ind[n] = count
                self.ind_to_predicates[names[0]] = count
                count += 1

        # print('num relations',len(self.ind_to_predicates))
        self.ind_to_classes = sorted(self.ind_to_classes, key=lambda k: self.ind_to_classes[k])
        self.ind_to_predicates = sorted(self.ind_to_predicates, key=lambda k: self.ind_to_predicates[k])

        self.filenames, keep = load_image_filenames(image_file)

        # self.roidb_file, self.mode, num_im, num_val_im = num_val_im,
        #     filter_empty_rels=filter_empty_rels,

        self.split_mask = \
            load_graphs(keep, self.roidb_file, self.filenames, self.num_classes,
                        self._class_to_ind, self._relation_to_ind, self._attribute_to_ind,
                        filter_empty_rels=filter_empty_rels, filter_non_overlap=filter_non_overlap,
                        mode =self.mode, num_im=num_im, num_val_im = num_val_im)
        self.split_mask=self.split_mask[keep]
        # self.split_mask = self.split_mask[:len(self.filenames)]
        # print('filename num', len(self.filenames))
        # print('split num', len(self.split_mask))
        self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]

        # graphs_file, filenames, num_classes, _class_to_ind, _relation_to_ind, _attribute_to_ind,
        #                 filter_empty_rels=True, filter_non_overlap=False , mode='train', num_im=-1, num_val_im=0

        if use_proposals:
            print("Loading proposals", flush=True)
            p_h5 = h5py.File(PROPOSAL_FN, 'r')
            rpn_rois = p_h5['rpn_rois']
            rpn_scores = p_h5['rpn_scores']
            rpn_im_to_roi_idx = np.array(p_h5['im_to_roi_idx'][self.split_mask])
            rpn_num_rois = np.array(p_h5['num_rois'][self.split_mask])

            self.rpn_rois = []
            for i in range(len(self.filenames)):
                rpn_i = np.column_stack((
                    rpn_scores[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                    rpn_rois[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                ))
                self.rpn_rois.append(rpn_i)
        else:
            self.rpn_rois = None

        # You could add data augmentation here. But we didn't.
        # tform = []
        # if self.is_train:
        #     tform.append(RandomOrder([
        #         Grayscale(),
        #         Brightness(),
        #         Contrast(),
        #         Sharpness(),
        #         Hue(),
        #     ]))

        tform = [
            SquarePad(),
            Resize(IM_SCALE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform_pipeline = Compose(tform)

    @property
    def coco(self):
        """
        :return: a Coco-like object that we can use to evaluate detection!
        """
        anns = []
        for i, (cls_array, box_array) in enumerate(zip(self.gt_classes, self.gt_boxes)):
            for cls, box in zip(cls_array.tolist(), box_array.tolist()):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': i,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'ayy lmao'},
            'images': [{'id': i} for i in range(self.__len__())],
            'categories': [{'supercategory': 'person',
                               'id': i, 'name': name} for i, name in enumerate(self.ind_to_classes) if name != '__background__'],
            'annotations': anns,
        }
        fauxcoco.createIndex()
        return fauxcoco

    @property
    def is_train(self):
        return self.mode.startswith('train')

    @classmethod
    def splits(cls, *args, **kwargs):
        """ Helper method to generate splits of the dataset"""
        train = cls('train', *args, **kwargs)
        val = cls('val', *args, **kwargs)
        test = cls('test', *args, **kwargs)
        return train, val, test

    @classmethod
    def load_vg_annotation(self, filename):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        #************************
        width, height = self._get_size(index)
        #************************
        tree = ET.parse(os.path.join(GRAPH_DIR,filename+'.xml'))
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        # Max of 16 attributes are observed in the data
        gt_attributes = np.zeros((num_objs, 16), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        obj_dict = {}
        ix = 0
        for obj in objs:
            obj_name = obj.find('name').text.lower().strip()
            if obj_name in self._class_to_ind:
                bbox = obj.find('bndbox')
                x1 = max(0, float(bbox.find('xmin').text))
                y1 = max(0, float(bbox.find('ymin').text))
                x2 = min(width - 1, float(bbox.find('xmax').text))
                y2 = min(height - 1, float(bbox.find('ymax').text))
                # If bboxes are not positive, just give whole image coords (there are a few examples)
                if x2 < x1 or y2 < y1:
                    print('Failed bbox in %s, object %s' % (filename, obj_name))
                    x1 = 0
                    y1 = 0
                    x2 = width - 1
                    y2 = width - 1
                cls = self._class_to_ind[obj_name]
                obj_dict[obj.find('object_id').text] = ix
                atts = obj.findall('attribute')
                n = 0
                for att in atts:
                    att = att.text.lower().strip()
                    if att in self._attribute_to_ind:
                        gt_attributes[ix, n] = self._attribute_to_ind[att]
                        n += 1
                    if n >= 16:
                        break
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
                ix += 1
        # clip gt_classes and gt_relations
        gt_classes = gt_classes[:ix]
        gt_attributes = gt_attributes[:ix, :]

        overlaps = scipy.sparse.csr_matrix(overlaps)
        gt_attributes = scipy.sparse.csr_matrix(gt_attributes)

        rels = tree.findall('relation')
        num_rels = len(rels)
        gt_relations = set()  # Avoid duplicates
        for rel in rels:
            pred = rel.find('predicate').text
            if pred:  # One is empty
                pred = pred.lower().strip()
                if pred in self._relation_to_ind:
                    try:
                        triple = []
                        triple.append(obj_dict[rel.find('subject_id').text])
                        triple.append(self._relation_to_ind[pred])
                        triple.append(obj_dict[rel.find('object_id').text])
                        gt_relations.add(tuple(triple))
                    except:
                        pass  # Object not in dictionary
        gt_relations = np.array(list(gt_relations), dtype=np.int32)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_attributes': gt_attributes,
                'gt_relations': gt_relations,
                'gt_overlaps': overlaps,
                'width': width,
                'height': height,
                'flipped': False,
                'seg_areas': seg_areas}

    def __getitem__(self, index):
        # annotation = self.load_vg_annotation(self.filenames[index])

        # Optionally flip the image if we're doing training
        # flipped = self.is_train and np.random.random() > 0.5
        # print('mode',self.mode)
        # print(self.filenames[index])
        filename = self.filenames[index]
        gt_boxes, gt_classes, _, gt_rels, gt_attrs = torch.load(os.path.join(VG_IMAGES,filename+'.pt'))
        path1 = os.path.join(DATA_PATH, 'VG_100K', filename + '.jpg')
        path2 = os.path.join(DATA_PATH, 'VG_100K_2', filename + '.jpg')
        if os.path.exists(path1):
            features = Image.open(path1)
            w,h = features.size
            features = features.convert('RGB')
        elif os.path.exists(path2):
            features = Image.open(path2)
            w, h = features.size
            features = features.convert('RGB')
        else:
            raise ValueError('Cannot find the image.')
        gt_rels = gt_rels[0].cpu().numpy()
        gt_rels = gt_rels[:,[0,2,1]]
        # gt_classes = torch.cat(((torch.zeros_like(gt_classes)+index).unsqueeze(1),gt_classes.unsqueeze(1)),dim=1)
        # gt_boxes = self.gt_boxes[index].copy()
        # print(gt_boxes)
        # Boxes are already at BOX_SCALE
        # if self.is_train:
        #     # crop boxes that are too large. This seems to be only a problem for image heights, but whatevs
        #     gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clip(
        #         None, BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[1])
        #     gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clip(
        #         None, BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[0])

            # # crop the image for data augmentation
            # image_unpadded, gt_boxes = random_crop(image_unpadded, gt_boxes, BOX_SCALE, round_boxes=True)

        # box_scale_factor = BOX_SCALE / max(w, h)

        # if flipped:
        #     scaled_w = int(box_scale_factor * float(w))
        #     # print("Scaled w is {}".format(scaled_w))
        #     image_unpadded = image_unpadded.transpose(Image.FLIP_LEFT_RIGHT)
        #     gt_boxes[:, [0, 2]] = scaled_w - gt_boxes[:, [2, 0]]

        img_scale_factor = IM_SCALE / max(w, h)
        if h > w:
            im_size = (IM_SCALE, int(w * img_scale_factor), img_scale_factor)
        elif h < w:
            im_size = (int(h * img_scale_factor), IM_SCALE, img_scale_factor)
        else:
            im_size = (IM_SCALE, IM_SCALE, img_scale_factor)

        # gt_rels = self.relationships[index].copy()

        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.mode == 'train'
            old_size = gt_rels.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_rels:
                all_rel_sets[(o0, o1)].append(r)
            gt_rels = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            gt_rels = np.array(gt_rels)

        entry = {
            'img': features.cpu(),
            'img_size': im_size,
            'gt_boxes': gt_boxes.cpu().numpy(),
            'gt_classes': gt_classes.cpu().numpy().astype(np.int32),
            'gt_relations': gt_rels.astype(np.int32),
            'scale': img_scale_factor,  # Multiply the boxes by this.
            'index': index,
            'flipped': False,
            'fn': self.filenames[index],
        }

        if self.rpn_rois is not None:
            entry['proposals'] = self.rpn_rois[index]

        assertion_checks(entry)
        return entry

    def __len__(self):
        return len(self.filenames)

    @property
    def num_predicates(self):
        return len(self.ind_to_predicates)

    @property
    def num_classes(self):
        return len(self.ind_to_classes)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MISC. HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def assertion_checks(entry):
    # im_size = tuple(entry['img'].size())
    # if len(im_size) != 3:
    #     raise ValueError("Img must be dim-3")
    #
    # c, h, w = entry['img'].size()
    # if c != 3:
    #     raise ValueError("Must have 3 color channels")

    num_gt = entry['gt_boxes'].shape[0]
    if entry['gt_classes'].shape[0] != num_gt:
        raise ValueError("GT classes and GT boxes must have same number of examples")

    assert (entry['gt_boxes'][:, 2] >= entry['gt_boxes'][:, 0]).all()
    assert (entry['gt_boxes'] >= -1).all()


def load_image_filenames(image_file,image_dir=VG_IMAGES):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    :param image_file: JSON file. Elements contain the param "image_id".
    :param image_dir: directory where the VisualGenome images are located
    :return: List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592', '1722', '4616', '4617']+torch.load('/share/yutong/projects/neural-motifs-train-Copy/remove_train.pt')+torch.load('/share/yutong/projects/neural-motifs-train-Copy/remove_test.pt')+torch.load('/share/yutong/projects/neural-motifs-train-Copy/remove_val.pt')
    # print(corrupted_ims)
    fns = []
    keep = []
    ind=0
    for i, img in enumerate(im_data):
        print('img',img)
        basename = str(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(image_dir, basename)
        filename_annotation = os.path.join(GRAPH_DIR, basename)
        if os.path.exists(filename+'.pt') and os.path.exists(filename_annotation+'.xml'):
            # print('filename',filename)
            fns.append(filename.split('/')[-1])
            keep.append(ind)
        ind+=1
    # print('im data',len(im_data))
    # print('fns',len(fns),'keep',len(keep))
    # assert len(fns) == 108073
    return fns, keep


def load_graphs(keep, graphs_file, filenames, num_classes, _class_to_ind, _relation_to_ind, _attribute_to_ind,
                filter_empty_rels=True, filter_non_overlap=False ,  mode='train', num_im=-1, num_val_im=0):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    # ************************
    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    roi_h5 = h5py.File(graphs_file, 'r')
    data_split = roi_h5['split'][:]
    split = 2 if mode == 'test' else 0
    split_mask = data_split == split
    split_mask = split_mask[keep]
    #
    # # Filter out images without bounding boxes
    # split_mask &= roi_h5['img_to_first_box'][:] >= 0
    # if filter_empty_rels:
    #     split_mask &= roi_h5['img_to_first_rel'][:] >= 0
    # split_mask = split_mask[:len(filenames)]
    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if mode == 'val':
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]


    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True
    # ************************

    # torch.save(remove,'remove_'+mode+'.pt')
    return split_mask

def load_split(graphs_file, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    roi_h5 = h5py.File(graphs_file, 'r')
    data_split = roi_h5['split'][:]
    split = 2 if mode == 'test' else 0
    split_mask = data_split == split

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if mode == 'val':
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]


    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True


    return split_mask


def load_info(info_file):
    """
    Loads the file containing the visual genome label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
    """
    info = json.load(open(info_file, 'r'))
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return ind_to_classes, ind_to_predicates


def vg_collate(data, num_gpus=3, is_train=False, mode='det'):
    assert mode in ('det', 'rel')
    blob = Blob(mode=mode, is_train=is_train, num_gpus=num_gpus,
                batch_size_per_gpu=len(data) // num_gpus)
    for d in data:
        blob.append(d)
    blob.reduce()
    return blob


class VGDataLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def splits(cls, train_data, val_data, batch_size=3, num_workers=1, num_gpus=3, mode='det',
               **kwargs):
        assert mode in ('det', 'rel')
        train_load = cls(
            dataset=train_data,
            batch_size=batch_size * num_gpus,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=True),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        val_load = cls(
            dataset=val_data,
            batch_size=batch_size * num_gpus if mode=='det' else num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=False),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        return train_load, val_load
