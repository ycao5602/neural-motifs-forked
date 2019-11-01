"""
Get counts of all of the examples in the dataset. Used for creating the baseline
dictionary model
"""

import numpy as np
from dataloaders.visual_genome import VG
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from lib.pytorch_misc import nonintersecting_2d_inds


def get_counts(train_data=VG(mode='train', filter_duplicate_rels=False, num_val_im=5000), must_overlap=True):
    """
    Get counts of all of the relations. Used for modeling directly P(rel | o1, o2)
    :param train_data: 
    :param must_overlap: 
    :return: 
    """
    fg_matrix = np.zeros((
        train_data.num_classes,
        train_data.num_classes,
        train_data.num_predicates,
    ), dtype=np.int64)

    bg_matrix = np.zeros((
        train_data.num_classes,
        train_data.num_classes,
    ), dtype=np.int64)

    # for ex_ind in range(len(train_data)):
    for i, data_dict in enumerate(train_data):
        # gt_classes = data_dict['gt_classes'][:,1].copy()
        gt_classes = data_dict['gt_classes'].copy()
        gt_relations =data_dict['gt_relations'].copy()
        gt_boxes = data_dict['gt_boxes'].copy()
        # print('gt_classes.size()',gt_classes)
        # print('gt_relations.size()', gt_relations)
        # print('gt_boxes.size()', gt_boxes)

        # For the foreground, we'll just look at everything
        # print('gt relations',gt_relations)
        o1o2 = gt_classes[gt_relations[:, :2]]%152
        for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
            # print(o1,o2,gtr)
            fg_matrix[o1, o2, gtr] += 1

        # For the background, get all of the things that overlap.
        # print(box_filter(gt_boxes, must_overlap=must_overlap))
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]%152
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float)) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

if __name__ == '__main__':
    fg, bg = get_counts(must_overlap=False)
