
from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
import torch

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
import dill as pkl
import os
import json

conf = ModelConfig()
if conf.model == 'motifnet':
    from lib.rel_model import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
else:
    raise ValueError()

info = json.load(open('/share/yutong/projects/neural-motifs/data/VG-SGG-dicts.json', 'r'))
info['label_to_idx']['__background__'] = 0
info['predicate_to_idx']['__background__'] = 0

class_to_ind = info['label_to_idx']
predicate_to_ind = info['predicate_to_idx']
ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

detector = RelModel(classes=ind_to_classes, rel_classes=ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=conf.order,
                    nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
                    use_proposals=conf.use_proposals,
                    pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
                    pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
                    pooling_dim=conf.pooling_dim,
                    rec_dropout=conf.rec_dropout,
                    use_bias=conf.use_bias,
                    use_tanh=conf.use_tanh,
                    limit_vision=conf.limit_vision
                    )


detector.cuda()
ckpt = torch.load(conf.ckpt)
print('loaded checkpoint')
optimistic_restore(detector, ckpt['state_dict'])
print('restored')
# if conf.mode == 'sgdet':
#     det_ckpt = torch.load('checkpoints/new_vgdet/vg-19.tar')['state_dict']
#     detector.detector.bbox_fc.weight.data.copy_(det_ckpt['bbox_fc.weight'])
#     detector.detector.bbox_fc.bias.data.copy_(det_ckpt['bbox_fc.bias'])
#     detector.detector.score_fc.weight.data.copy_(det_ckpt['score_fc.weight'])
#     detector.detector.score_fc.bias.data.copy_(det_ckpt['score_fc.bias'])

all_pred_entries = []
all_batches=[]
problem=[]
def val_batch(batch_num, b, thrs=(20, 50, 100)):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]
    # print('det res: ',det_res)

    for i, ((boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i),_, obj_dists) in enumerate(det_res):
        # gt_entry = {
        #     'gt_classes': val.gt_classes[batch_num + i].copy(),
        #     'gt_relations': val.relationships[batch_num + i].copy(),
        #     'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        # }
        # assert np.all(objs_i[rels_i[:,0]] > 0) and np.all(objs_i[rels_i[:,1]] > 0)
        # # assert np.all(rels_i[:,2] > 0)
        # print(obj_dists.size())
        print('rels_i',torch.Tensor(rels_i).size())
        print('pred_scores_i',torch.max(torch.Tensor(pred_scores_i), 1)[1])
        print('pred_scores_i size', torch.Tensor(pred_scores_i))
        print('boxes_i', torch.Tensor(boxes_i).size())
        print('boxes_i',boxes_i)
        pred_entry = {
            #'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            #'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,
        }
        all_pred_entries.append(pred_entry)


# evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=conf.multi_pred)
if conf.cache is not None and os.path.exists(conf.cache):
    print("Found {}! Loading from it".format(conf.cache))
    with open(conf.cache,'rb') as f:
        all_pred_entries = pkl.load(f)
    for i, pred_entry in enumerate(tqdm(all_pred_entries)):
        gt_entry = {
            'gt_classes': val.gt_classes[i].copy(),
            'gt_relations': val.relationships[i].copy(),
            'gt_boxes': val.gt_boxes[i].copy(),
        }

else:
    detector.eval()
    print('len val',len(val_loader))
    for val_b, batch in enumerate(tqdm(val_loader)):
        # print('val_b',val_b)
        # if val_b>10:
        #     break
        # print('batch.ids',batch.ids )
        # continue
        # try:
        all_batches.extend(batch.ids)
        print('batch.ids',batch.ids)
        val_batch(conf.num_gpus*val_b, batch)
        # except:
        #     problem.extend(batch.ids)
        #     print('id ',batch.ids,' went wrong')
        #     pass
        if val_b%1000==0 and not val_b==0:
            print('saving for batch: ',val_b)
            predictions = dict(zip(all_batches, all_pred_entries))
            torch.save(predictions, 'img_sg_val_'+str(val_b)+'.pt')
            all_pred_entries=[]
            all_batches=[]

    predictions = dict(zip(all_batches, all_pred_entries))
    # torch.save(predictions, 'img_sg_val_' + str(val_b) + '.pt')
    # torch.save(problem,'problem.pt')