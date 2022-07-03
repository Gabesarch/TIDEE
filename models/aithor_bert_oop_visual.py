from ai2thor.controller import Controller
import os
import ipdb
st = ipdb.set_trace
import numpy as np
import random
import cv2
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import sys
import numpy as np
import torch
import time
from utils.improc import *
import torch.nn.functional as F 
from arguments import args
from nets.solq import DDETR
import SOLQ.util.misc as ddetr_utils
from SOLQ.util import box_ops
from backend import saverloader
from nets.relationsnet_visual import OOPNet
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from Detection_Metrics.pascalvoc_nofiles import get_map, ValidateFormats, ValidateCoordinatesTypes, add_bounding_box
import glob
from Detection_Metrics.lib.BoundingBox import BoundingBox
from Detection_Metrics.lib.BoundingBoxes import BoundingBoxes
from Detection_Metrics.lib.Evaluator import *
from Detection_Metrics.lib.utils_pascal import BBFormat
import torch.nn as nn
from tqdm import tqdm
from models.aithor_bert_oop_visual_base import Ai2Thor_Base
import pickle

def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out

class Ai2Thor(Ai2Thor_Base):
    def __init__(self):   

        super(Ai2Thor, self).__init__()

        # num_classes = 2 # out of place or not

        if args.do_predict_oop:
            num_classes = len(self.include_classes) 
            self.label_to_id = {False:0, True:1} # not out of place = False
            self.id_to_label = {0:'IP', 1:'OOP'}
            num_classes2 = 2 # 0 oop + 1 not oop + 2 no object
            # for inference
            self.score_boxes_name = 'pred1'
            self.score_labels_name = 'pred1' # semantic labels
            self.score_labels_name1 = 'pred1' # semantic labels
            self.score_labels_name2 = 'pred2' # which prediction head has the OOP label?

            self.score_threshold_oop = args.score_threshold_oop
            self.score_threshold = args.score_threshold_cat
        else:
            assert(False) # need two head detector (semantic and OOP)

        if args.do_visual_and_language_oop:
            print("LAGUAGE + VISUAL")
        elif args.do_visual_only_oop:
            print("VISUAL ONLY")
        elif args.do_language_only_oop:
            print("LANGUAGE ONLY")
        else:
            assert(False)

        load_pretrained = False
        self.ddetr = DDETR(num_classes, load_pretrained, num_classes2=num_classes2).cuda()
        

        _ = saverloader.load_from_path(args.SOLQ_oop_checkpoint, self.ddetr, None, strict=True)
        self.ddetr.eval() #.cpu()

        self.model = OOPNet(num_classes=num_classes2)
        self.model.cuda()

        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        params_to_optimize = self.model.parameters()
        # self.optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
        #                               weight_decay=args.weight_decay)
        self.optimizer = AdamW(params_to_optimize,
                    lr = args.lr_vboop, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
        # lr_drop = args.lr_drop # every X epochs, drop lr by 0.1
        self.lr_scheduler = None #torch.optim.lr_scheduler.StepLR(self.optimizer, lr_drop)

        if args.freeze_layers>0:
            freeze_layers = list(np.arange(args.freeze_layers))
            freeze_layers = ['layer.'+str(freeze_layers_i)+'.' for freeze_layers_i in freeze_layers]
            for name, param in self.model.named_parameters():
                if 'embeddings' in name:
                    param.requires_grad = False
                for fl in freeze_layers:
                    if fl in name:
                        param.requires_grad = False

        for name, param in self.model.named_parameters():
            print(name, "requires grad?", param.requires_grad)

        self.start_step = 1
        if args.load_model:
            path = args.load_model_path 

            if args.lr_scheduler_from_scratch:
                print("LR SCHEDULER FROM SCRATCH")
                lr_scheduler_load = None
            else:
                lr_scheduler_load = self.lr_scheduler

            if args.optimizer_from_scratch:
                print("OPTIMIZER FROM SCRATCH")
                optimizer_load = None
            else:
                optimizer_load = self.optimizer
            if True: 
                checkpoint = torch.load(path)
                model_dict = self.model.state_dict()
                pretrained_dict = checkpoint['model_state_dict']
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if "classifier.0.weight" not in k}
                model_dict.update(pretrained_dict) 
                msg = self.model.load_state_dict(pretrained_dict, strict=False)
                print(msg)
            else:
                self.start_step = saverloader.load_from_path(path, self.model, optimizer_load, strict=(not args.load_strict_false), lr_scheduler=lr_scheduler_load)

        if args.start_one:
            self.start_step = 1

        self.max_iters = args.max_iters
        self.log_freq = args.log_freq

    def run_episodes(self):
        
        self.ep_idx = 0

        if args.eval_test_set:
            self.eval_on_test()
            return 

        if args.run_data_ordered:
            self.n_iter = iter(list(np.arange(args.n_train)))
            self.mapnames_iter = iter(self.mapnames_train)
            self.mapname_current = next(self.mapnames_iter)
        
        for iter_ in range(self.start_step, self.max_iters+1):

            print("Begin iter", iter_)
            print("set name:", args.set_name)
            # print("mode:", args.transformer_mode)

            self.summ_writer = utils.improc.Summ_writer(
                writer=self.writer,
                global_step=iter_,
                log_freq=self.log_freq,
                fps=8,
                just_gif=True)
            
            total_loss = self.run_train()

            if total_loss is not None:
                self.optimizer.zero_grad()
                total_loss.backward()
                if args.clip_max_norm > 0:
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_max_norm)
                else:
                    grad_total_norm = ddetr_utils.get_total_grad_norm(self.model.parameters(), args.clip_max_norm)
                self.optimizer.step()

            print(f"loss for iter {iter_} is: {total_loss}")

            if args.run_val:
                if iter_ % args.val_freq == 0:
                    with torch.no_grad():
                        self.run_val()
                        
            if self.lr_scheduler is not None:
                if iter_ % args.lr_scheduler_freq== 0:
                    self.lr_scheduler.step()

            if iter_ % args.save_freq == 0:
                saverloader.save_checkpoint(self.model, self.checkpoint_path, iter_, None, keep_latest=args.keep_latest, lr_scheduler=self.lr_scheduler)

        self.controller.stop()
        time.sleep(1)

    def run_train(self):
        total_loss = torch.tensor(0.0).cuda() #to(self.device0) #.cuda()

        if args.run_data_ordered:
            n = next(self.n_iter, None)
            if n is None:
                self.mapname_current = next(self.mapnames_iter, None)
                if self.mapname_current is None:
                    self.mapnames_iter = iter(self.mapnames_train)
                    self.mapname_current = next(self.mapnames_iter, None)
                self.n_iter = iter(list(np.arange(args.n_train)))
                n = next(self.n_iter, None)
            if args.mess_up_from_loaded:
                n_messup = n
            else:
                n_messup = None
            mapname = self.mapname_current
        else:
            rand = np.random.randint(len(self.mapnames_train))
            mapname = self.mapnames_train[rand]
            n = np.random.randint(args.n_train)
            if args.mess_up_from_loaded:
                n_messup = np.random.randint(args.n_train_messup)
            else:
                n_messup = None
        
        if args.load_train_agent:
            outputs = self.load_agent_output(n, mapname, args.train_load_dir, n_messup=n_messup, save=True)
        else:
            # n = np.random.randint(args.max_load_n)
            # n_messup = np.random.randint(args.n_train_messup)
            outputs = self.load_agent_output(None, mapname, None, n_messup=n_messup, run_agent=True)
        
        targets = outputs[1]

        loss, probs = self.model(targets['bert_token_ids'], targets['bert_attention_masks'], targets['ddetr_features'], targets['labels2'])
        total_loss += loss

        targets['probs'] = probs

        self.summ_writer.summ_scalar('train/total_loss', total_loss.cpu().item())
        
        # if self.summ_writer.save_this:
        # self.summ_writer.save_this = True
        if False: #self.summ_writer.save_this:
            batch_inds = targets['batch_lens']
            # first one only
            batch_inds_0 = batch_inds==0
            probs_0 = torch.argmax(targets['probs'][batch_inds_0], dim=1).cpu().numpy()
            scores_0 = torch.max(targets['probs'][batch_inds_0], dim=1).values.cpu().numpy()
            pred_labels_0 = [self.id_to_label[probs_i] for probs_i in list(probs_0)]
            rgb_0 = targets['rgb'][0].cpu().numpy()
            boxes_0 = targets['boxes'][batch_inds_0].cpu().numpy()
            labels_0 = targets['labels2'][batch_inds_0].cpu().numpy()
            labels_0 = [self.id_to_label[labels_0_i] for labels_0_i in list(labels_0)]
            img_torch = self.draw_ddetr_predictions(rgb_0, boxes_0, pred_labels_0, scores_0, labels_0)

            img_torch = F.interpolate(img_torch.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)
            rgb_x_vis = img_torch.unsqueeze(0) - 0.5

            name = f'train/detections_{mapname}'
            self.summ_writer.summ_rgb(name, rgb_x_vis)

            argmax_p = torch.argmax(probs, dim=1)
            correct = targets['labels2']==argmax_p
            accuracy = torch.sum(correct)/len(targets['labels2'])
            self.summ_writer.summ_scalar('train/accuracy', accuracy.cpu().item())
            which_inplace = targets['labels2']==0
            correct_ip = correct[which_inplace]
            accuracy_in_place = torch.sum(correct_ip)/len(correct_ip)
            self.summ_writer.summ_scalar('train/accuracy_in_place', accuracy_in_place.cpu().item())
            which_outplace = targets['labels2']==1
            correct_oop = correct[which_outplace]
            accuracy_out_place = torch.sum(correct_oop)/len(correct_oop)
            self.summ_writer.summ_scalar('train/accuracy_out_place', accuracy_out_place.cpu().item())
            AP = average_precision_score(targets['labels2'].cpu().numpy(), probs[:,0].cpu().numpy())
            self.summ_writer.summ_scalar('train/AP', torch.from_numpy(np.array(AP)))

                
        return total_loss

    def load_agent_output(self, n, mapname, load_dir, n_messup, pick_rand_n=False, always_load_all_samples=False, save=False, run_agent=False, override_existing=False):
        
        if not run_agent:
            root = os.path.join(load_dir, mapname) 
            if not os.path.exists(root):
                os.mkdir(root)
            
            if pick_rand_n or n is None:
                ns = os.listdir(root)
                ns = np.array([int(n_[0]) for n_ in ns])
                n = np.random.choice(ns)
        
            print(f"Chose n to be {n}")

            pickle_fname = f'{n}.p'
            fname_ = os.path.join(root, pickle_fname)
        else:
            fname_ = ""

        if not os.path.isfile(fname_) or override_existing or run_agent: # if doesn't exist generate it
            if override_existing:
                print("WARNING: OVERRIDING EXISTING FILE IF THERE IS ONE")
            else:
                print("file doesn not exist...generating it")
        
            self.controller.reset(scene=mapname)

            # load_dict = None

            if args.mess_up_from_loaded:
                messup_fname = os.path.join(args.mess_up_dir, mapname, f'{n_messup}.p')
                with open(messup_fname, 'rb') as f:
                    load_dict = pickle.load(f)
            else:
                load_dict = None

            if args.load_object_tracker:
                n_obs = n
                root = os.path.join(args.mapping_obs_dir, mapname) 
                pickle_fname = f'{n_obs}.p'
                fname = os.path.join(root, pickle_fname)
                with open(fname, 'rb') as f:
                    obs_dict = pickle.load(f)
            else:
                obs_dict = None

            outputs = self.run_agent(load_dict=load_dict, obs_dict=obs_dict)

            if save:
                print("saving", fname_)
                with open(fname_, 'wb') as f:
                    pickle.dump(outputs, f, protocol=4)
                print("done.")
        
        else:
            print("-----------------")
            print("LOADING OUTPUTS")
            print("-----------------")
            with open(fname_, 'rb') as f:
                outputs = pickle.load(f)        

        rgb = outputs[0]
        targets_ = outputs[1]
        if len(outputs[0])<args.batch_size:
            print("Warning: requested batch size larger than saved batch size. Returning max batch size of data.")
        if len(outputs[0])>args.batch_size and not always_load_all_samples:
            # sample
            N = np.arange(len(rgb))
            idxs = np.random.choice(N, size=args.batch_size, replace=False)
            rgb = rgb[idxs]
            targets = [targets[i] for i in list(idxs)]

        # concat and put on cuda
        count = 0
        batch_inds = torch.tensor([])
        targets = {}
        for out_ in targets_:
            for k in out_.keys():
                if k=='aithor_metadata':
                    continue
                if k=='relations':
                    if args.do_visual_only_oop:
                        if count==0:
                            targets['bert_token_ids'] = torch.tensor([])
                            targets['bert_attention_masks'] = torch.tensor([])
                        continue
                    # prepare for BERT
                    if args.do_GT_relations:
                        relations_ = out_['relations']['GT']
                    else:
                        relations_ = out_['relations']['pred']
                    # print(out_['relations']['GT'])
                    # print(out_['relations']['pred'])
                    # print(out_['labels2'])
                    # relations_ = out_['relations']
                    bert_token_ids_batch = torch.tensor([]).long()
                    bert_attention_masks_batch = torch.tensor([]).long()
                    for relations in relations_:
                        sentences_formatted = []
                        for relation_sentence in relations:
                            sentence_full = relation_sentence.replace('-', ' ')
                            sentences_formatted.append(sentence_full)
                        paragraph = '. '.join(sentence for sentence in sentences_formatted)
                        encoded_dict = self.bert_tokenizer.encode_plus(
                            paragraph,                      # paragraph to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 512,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                        )
                        bert_token_ids_batch = torch.cat([bert_token_ids_batch, encoded_dict['input_ids']], dim=0)
                        bert_attention_masks_batch = torch.cat([bert_attention_masks_batch, encoded_dict['attention_mask']], dim=0)
                    if count==0:
                        targets['bert_token_ids'] = bert_token_ids_batch
                        targets['bert_attention_masks'] = bert_attention_masks_batch
                    else:
                        targets['bert_token_ids'] = torch.cat([targets['bert_token_ids'], bert_token_ids_batch], dim=0)
                        targets['bert_attention_masks'] = torch.cat([targets['bert_attention_masks'], bert_attention_masks_batch], dim=0)
                elif count==0:
                    targets[k] = out_[k]
                else:
                    targets[k] = torch.cat([targets[k], out_[k]], dim=0)
            batch_inds = torch.cat([batch_inds, torch.ones(len(targets['labels2']))*count], dim=0)
            count += 1

        # for t in targets:
        for k in targets.keys():
            targets[k] = targets[k].cuda()
        rgb = rgb.cuda()

        outputs[0] = rgb
        outputs[1] = targets
        outputs.append(batch_inds)
        
        return outputs

    def run_val(self):

        print("VAL MODE")
        self.model.eval()

        total_loss = torch.tensor(0.0).cuda()
        
        # mapper = mAP(score_boxes_name=self.score_boxes_name, score_labels_name=self.score_labels_name)
        # if args.two_pred_heads:
        #     mapper2 = mAP(score_boxes_name=self.score_boxes_name, score_labels_name=self.score_labels_name2)

        num_total_iters = len(self.mapnames_val)*args.n_val
        for n in range(args.n_val):
            for episode in range(len(self.mapnames_val)):
                print("STARTING EPISODE ", episode, "iteration", n)

                mapname = self.mapnames_val[episode]
                print("MAPNAME=", mapname)

                if args.load_val_agent:
                    outputs = self.load_agent_output(n, mapname, args.val_load_dir, n_messup=n, save=True)
                else:
                    outputs = self.load_agent_output(None, mapname, None, n_messup=n, run_agent=True)

                targets = outputs[1]

                loss, probs = self.model(targets['bert_token_ids'], targets['bert_attention_masks'], targets['ddetr_features'], targets['labels2'])
                total_loss += loss

                targets['probs'] = probs

        self.summ_writer.summ_scalar('val/total_loss', total_loss.cpu().item())
        
        self.model.train()

    def save_output(self):

        # mapnames = self.mapnames_train + self.mapnames_val + self.mapnames_test
        mapnames = self.mapnames_test

        # mapnames = mapnames[:80]
        # mapnames = mapnames[40:80]
        # mapnames = mapnames[80:]

        print(mapnames)
        print("LENGTH MAPNAMES", len(mapnames))

        for n in range(args.n_test):
            for episode in range(len(mapnames)):
                mapname = mapnames[episode]

                print("MAPNAME=", mapname, "n=", n)

                # outputs = self.load_agent_output(n, mapname, args.bv_load_dir)
                outputs = self.load_agent_output(n, mapname, args.val_load_dir, n_messup=n, save=True)

                print("DONE.", "MAPNAME=", mapname, "n=", n)

    def eval_on_test(self):

        self.model.eval()

        errors = []

        gtFormat = ValidateFormats('xyxy', '-gtformat', errors)
        detFormat = ValidateFormats('xyxy', '-detformat', errors)
        allBoundingBoxes = BoundingBoxes()
        allClasses = []

        gtCoordType = ValidateCoordinatesTypes('abs', '-gtCoordinates', errors)
        detCoordType = ValidateCoordinatesTypes('abs', '-detCoordinates', errors)
        imgSize = (0, 0)
        if gtCoordType == CoordinatesType.Relative:  # Image size is required
            assert(False) #imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
        if detCoordType == CoordinatesType.Relative:  # Image size is required
            assert(False) #imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)
        
        # targets_all = []
        # files = os.listdir(args.save_dir_test_set)
        # num_files = len(files)
        # print("NUMBER OF FILES", num_files)
        visibilities = []
        labels_all =[]
        for n2 in range(args.n_test):
            for n in range(args.n_test_messup):
                for episode in range(len(self.mapnames_test)):

                    mapname = self.mapnames_test[episode]

                    print(f"n2={n2}_n={n}_episode={episode}")

                    # if n2==0 and n==0 and episode==12:
                    #     outputs = self.load_agent_output(n, mapname, args.test_load_dir, n_messup=n, save=True,override_existing=True)

                    # try:
                    outputs = self.load_agent_output(n, mapname, args.test_load_dir, n_messup=n, save=True)
                    # except:
                    #     continue
                    # continue
                    


                    targets = outputs[1]
                    batch_inds = outputs[2]

                    labels = targets['labels2']

                    ip_percent = torch.sum(labels==0)/len(labels)
                    oop_percent = torch.sum(labels==1)/len(labels)
                    # print(f"percent in place {ip_percent}")
                    # print(f"percent out of place {oop_percent}")
                    # print(torch.mean(targets['visibility'][labels==1])/torch.mean(targets['visibility'][labels==0]))
                    
                    labels_all.append(labels)
                    visibilities.append(targets['visibility'])

                    with torch.no_grad():
                        loss, probs = self.model(targets['bert_token_ids'], targets['bert_attention_masks'], targets['ddetr_features'], None)
                    # total_loss += loss

                    targets['probs'] = probs

                    pred_boxes = targets['pred_boxes'].cpu().numpy()
                    for z in range(pred_boxes.shape[0]):
                        prob = probs[z]
                        prob_argmax = int(torch.argmax(prob).cpu().numpy())
                        score = prob[prob_argmax].cpu().numpy()
                        pred_box = pred_boxes[z]
                        pred_label = self.id_to_label[prob_argmax]
                        # score = pred_scores[z]
                        bbox_params_det = [pred_label, score, pred_box[0], pred_box[1], pred_box[2], pred_box[3]]
                        allBoundingBoxes, allClasses = add_bounding_box(
                            bbox_params_det, 
                            allBoundingBoxes, 
                            allClasses, 
                            nameOfImage=int(str(episode)+str(n)+str(n2)), 
                            isGT=False, 
                            imgSize=imgSize, 
                            Format=detFormat, 
                            CoordType=detCoordType
                            )

                    gt_boxes = targets['boxes']
                    gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
                    gt_boxes[:,[0,2]] = gt_boxes[:,[0,2]] * self.W
                    gt_boxes[:,[1,3]] = gt_boxes[:,[1,3]] * self.H
                    gt_boxes = gt_boxes.cpu().numpy()
                    gt_labels = targets['labels2'].cpu().numpy()
                    for z in range(gt_boxes.shape[0]):
                        gt_box = gt_boxes[z]
                        bbox_params_gt = [self.id_to_label[int(gt_labels[z])], gt_box[0], gt_box[1], gt_box[2], gt_box[3]]
                        allBoundingBoxes, allClasses = add_bounding_box(
                            bbox_params_gt, 
                            allBoundingBoxes, 
                            allClasses, 
                            nameOfImage=int(str(episode)+str(n)+str(n2)), 
                            isGT=True, 
                            imgSize=imgSize, 
                            Format=gtFormat, 
                            CoordType=gtCoordType
                            )
        
        # Check visibility of objects
        visibilities = torch.cat(visibilities, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        ip_percent = torch.sum(labels_all==0)/len(labels_all)
        oop_percent = torch.sum(labels_all==1)/len(labels_all)
        print("STATS entire test set:")
        print(f"percent in place {ip_percent}")
        print(f"percent out of place {oop_percent}")
        print(torch.mean(visibilities[labels_all==1])/torch.mean(visibilities[labels_all==0]))
            
        for iou_thresh in [0.25, 0.5, 0.75]:
            mAP, classes, aps, ars, mAR, precision, recall = get_map(allBoundingBoxes, allClasses, do_st_=False, IOU_threshold=iou_thresh)

            print(f"mAP@{iou_thresh}:", mAP)
            print(f"aps@{iou_thresh}", aps)
            print(f"precision@{iou_thresh}", precision)
            print(f"recall@{iou_thresh}", recall)
        st()

    def draw_ddetr_predictions(self, img_, pred_boxes, pred_labels, scores, true_boxes, true_labels, score_threshold=0.5, text_size=1, rect_th=1, text_th=1):

        img = np.float32(img_).copy()
        score = 0.0
        for i in range(len(pred_boxes)):
            
            pred_class_name = pred_labels[i]

            score = scores[i]

            cv2.rectangle(img, (int(pred_boxes[i][0]), int(pred_boxes[i][1])), (int(pred_boxes[i][2]), int(pred_boxes[i][3])),(0, 255, 0), rect_th)
            cv2.putText(img,pred_class_name, (int(pred_boxes[i][0:1]), int(pred_boxes[i][1:2])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
            # cv2.putText(img,str(score), (int(pred_boxes[i][2:3]), int(pred_boxes[i][3:4])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)


        img2 = np.float32(img_).copy()
        score = 0.0
        for i in range(len(true_boxes)):
            
            pred_class_name = true_labels[i]

            cv2.rectangle(img2, (int(true_boxes[i][0]), int(true_boxes[i][1])), (int(true_boxes[i][2]), int(true_boxes[i][3])),(0, 255, 0), rect_th)
            cv2.putText(img2,pred_class_name, (int(true_boxes[i][0:1]), int(true_boxes[i][1:2])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

        img = torch.from_numpy(img).cuda()/255.
        img = img.permute(2,0,1).float()

        img2 = torch.from_numpy(img2).cuda()/255.
        img2 = img2.permute(2,0,1).float()

        img_torch = torch.cat([img, img2], dim=2)
        return img_torch

    

if __name__ == '__main__':
    Ai2Thor()


