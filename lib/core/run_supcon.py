import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
import os
from tqdm import tqdm

from utils import total_acc_cal, each_cls_acc_cal, seed_everything
from backbone import *
from loss import create_ce_loss, create_supcon_loss

class Model_SupCon(object):
    def __init__(self, cfg, dataloader_dict, logger):
        self.cfg = cfg
        self.training_opt = self.cfg["TRAINING_OPT"]
        self.dataloader_dict = dataloader_dict
        self.logger = logger

        self.mode = self.cfg["MODE"]
        self.logger.info("*************************[{}]: {} ({})***********************".format(self.cfg["TYPE"], self.cfg["EXP_TYPE"], self.mode))

        self.device = torch.device(self.training_opt["DEVICE"] if torch.cuda.is_available() else "cpu")  # specify the GPU.
        self.logger.info("===> Using {} GPU\n".format(self.device))

        self.model_dir = self.cfg["MODEL_DIR"]
        
        self._init_models()  
        self._init_optimizers() 
        self._init_criterions() 

        self.do_batch_shuffle = self.cfg["DATALOADER"]["SAMPLER"]["DO_BATCH_SHUFFLE"]
        
        self.epoch_steps = None
        if (self.cfg["DATALOADER"]["SAMPLER"]["SAMPLER_CLASS"] is not None and
            self.cfg["DATALOADER"]["SAMPLER"]["USE_EPOCH_STEPS"]):
            self.logger.info("-----------As using custom sampler, using steps for training----------")
            self.training_data_num = len(self.dataloader_dict["train"].dataset)
            self.epoch_steps = int(self.training_data_num / 
                                   self.cfg['DATALOADER']['BATCH_SIZE'])
            self.logger.info("epoch steps: {}".format(self.epoch_steps))

        self.lamda = self.cfg["LAMDA"]


    def _init_models(self,):
        self.logger.info("-----------------------init models--------------------")
        
        self.networks_defs = self.cfg["NETWORKS"]
        self.networks = {}
      
        for key, val in self.networks_defs.items():
            model_args = val["PARAMS"]
            self.networks[key] = (eval(val["MODEL_CREATE_FUNC"])(logger=self.logger, **model_args)).to(self.device)

        
        if self.model_dir is not None and self.mode in ["train_linear", "fine_tune"]:
            self._load_model()
        
       
        self._set_requires_grad()
        


    def _load_model(self,):
        self.logger.info('===> Loading pertrained model from: {}'.format(self.model_dir))

        checkpoint = torch.load(self.model_dir)
        pretrained_dict = checkpoint["state_dict_best"]

        for key, model in self.networks.items():
            if ((self.mode in ["train_linear", "fine_tune"] and key != "FEAT_MODEL") 
                or (self.mode == "eval_linear" and key == "PROJECTION_HEAD")):
                self.logger.info('===> Skipping {} loading'.format(key))
                continue
            
            self.logger.info('===> {} loading'.format(key))
            weights = pretrained_dict[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            x = model.state_dict()
            x.update(weights)
            model.load_state_dict(x)
    


    def _set_requires_grad(self,):
        for key, val in self.networks_defs.items():
            if "FIX" in val and val["FIX"]:
                self.logger.info("===> Freezing {} model weights".format(key))
                for param_name, param in self.networks[key].named_parameters():
                    param.requires_grad = False


    def _init_optimizers(self,):
        self.logger.info("--------------Initializing model optimizer and scheduler--------------")
        self.model_optim_params_list = []
        for key, val in self.networks_defs.items():
            optim_params = val["OPTIM_PARAMS"]
            self.model_optim_params_list.append({'params': self.networks[key].parameters(), 'lr': optim_params['lr'], \
                                                 'betas': (optim_params['beta1'], optim_params['beta2']), \
                                                 'weight_decay': optim_params['weight_decay']})
        self.logger.info("===> Using Adam optimizer.")
        self.model_optimizer = optim.Adam(self.model_optim_params_list)

        self.model_optimizer_scheduler = None
        if self.training_opt["SCHEDULER"]["USE_SCHEDULER"]:
            if self.training_opt["SCHEDULER"]["COSLR_ENDLR"] is not None:
                coslr_endlr = self.training_opt["SCHEDULER"]["COSLR_ENDLR"]
                self.logger.info("===> Using coslr, eta_min={}".format(coslr_endlr))
                self.model_optimizer_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer, self.training_opt["NUM_EPOCHS"], eta_min=coslr_endlr)
            if self.training_opt["SCHEDULER"]["STEPLR_PARAMS"] is not None:
                self.scheduler_params = self.training_opt["SCHEDULER"]["STEPLR_PARAMS"]
                step_size = self.scheduler_params["step_size"]
                gamma = self.scheduler_params["gamma"]
                self.logger.info("===> Using steplr, step_size={}, gamma={}".format(step_size, gamma))
                self.model_optimizer_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, step_size=step_size, gamma=gamma)



    def _init_criterions(self,):
        self.criterions = {}
        self.logger.info("\n----------------------Initializing loss function----------------------")
        crtierion_defs = self.cfg["CRITERIONS"]

        for key, val in crtierion_defs.items():
            param_dict = val["PARAM_DICT"]
            if param_dict is None:
                self.criterions[key] = eval(val["LOSS_CREATE_FUNC"])(logger=self.logger, device=self.device)
            else:
                self.criterions[key] = eval(val["LOSS_CREATE_FUNC"])(logger=self.logger, device=self.device, **param_dict)


    def get_model_file_path(self,):
        output_dir = self.cfg["OUTPUT_DIR"]
        exp_data_name = self.cfg["DATASET_NAME"]  # name of the dataset used for the experiment
        dataset_desc = self.cfg["DATASET_DESC"]  # Description of the dataset
        if (self.cfg["IMBALANCED"]):
            imb_desc = self.cfg["IMB_DESC"]
            model_save_dir = os.path.join(output_dir, exp_data_name, dataset_desc, "imbalanced", imb_desc, "models")
        else:
            model_save_dir = os.path.join(output_dir, exp_data_name, dataset_desc, "balanced", "models")
        
        type = self.cfg["TYPE"]
        model_save_dir = os.path.join(model_save_dir, type, self.cfg["RUN_DESC"],
                                                           self.cfg["EXP_DESC"])
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        model_name = "{}_{}.pth".format(self.cfg["EXP_TYPE"], self.cfg["MODE"])
        model_file = os.path.join(model_save_dir, model_name)
        
        return model_file


    def save_model(self, epoch, best_epoch, best_model_weights, best_acc):
        model_states = {'epoch': epoch,
                        'best_epoch': best_epoch,
                        'state_dict_best': best_model_weights,
                        'best_acc': best_acc}
        
        # save model
        model_file_path = self.get_model_file_path()

        if os.path.exists(model_file_path):
            os.remove(model_file_path)
        torch.save(model_states, model_file_path)

        return model_file_path


    def reset_model(self, model_state):
        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            model.load_state_dict(weights)


    def shuffle_batch(self, inputs, labels, transformed1, transformed2):
        index = torch.randperm(inputs.size(0))
        inputs = inputs[index]
        labels = labels[index]
        transformed1 = transformed1[index]
        transformed2 = transformed2[index]

        return inputs, labels, transformed1, transformed2


    def train(self,):
        self.logger.info("\n---------------------- ****** Model train ****** ---------------------")
        best_model_weights = {}
        for key in list(self.cfg["NETWORKS"].keys()):
            best_model_weights[key] = copy.deepcopy(self.networks[key].state_dict())
        best_acc = 0.0
        best_epoch = 0
        end_epoch = self.training_opt["NUM_EPOCHS"]
        
        self.logger.info("===> lamda: {}".format(self.lamda))
        self.logger.info("===> no projection head")
        self.logger.info("===> no use aug for classifier")

        for epoch in range(1, end_epoch + 1):
            self.logger.info('===> Epoch: [{}/{}]'.format(epoch, end_epoch))

            for model in self.networks.values():
                model.train()
            torch.cuda.empty_cache()

            train_total_loss = []  
            train_supcon_loss = []
            train_ce_loss = []                                                    # list of each mini-batch loss
            train_total_preds = torch.empty(0, dtype=torch.long).to(self.device)  # predict labels of total test set
            train_total_labels = torch.empty(0, dtype=torch.long).to(self.device) # true labels of total test set

            for step, (data, target, data_transformed_list) in enumerate(tqdm(self.dataloader_dict["train"])):
               
                if self.epoch_steps is not None:
                    if step == self.epoch_steps:
                        break
                
                data, target = data.float().to(self.device), target.long().to(self.device)  
                
           
                data_transformed1, data_transformed2 = data_transformed_list[0].float().to(self.device), \
                                                       data_transformed_list[1].float().to(self.device)
          
                self.model_optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    if self.mode == "train_supcon":
                        feature1 = self.networks["FEAT_MODEL"](data_transformed1)
                        feature2 = self.networks["FEAT_MODEL"](data_transformed2)
                        
                        feature1 = F.normalize(feature1, dim=1)
                        feature2 = F.normalize(feature2, dim=1)
                    
                    feature = self.networks["FEAT_MODEL"](data)
                    logit = self.networks["CLASSIFIER"](feature)  

                    if self.mode == "train_supcon":
                        projection_feature = torch.cat([feature1.unsqueeze(1), feature2.unsqueeze(1)], dim=1)              
                        sup_loss = self.criterions["CL_LOSS"](projection_feature, target)
 
                    ce_loss = self.criterions["CE_LOSS"](logit, target)

                    output = F.softmax(logit, 1)
                    _, preds = output.max(dim=1)
                    train_total_preds = torch.cat((train_total_preds, preds))
                    train_total_labels = torch.cat((train_total_labels, target))

                    if self.mode == "train_supcon":
                        total_loss = ce_loss + self.lamda * sup_loss
                    else:
                        total_loss = ce_loss
                    
                    train_supcon_loss.append(sup_loss.item())
                    train_ce_loss.append(ce_loss.item())
                    train_total_loss.append(total_loss.item())

                    total_loss.backward()
                    self.model_optimizer.step()

            if self.model_optimizer_scheduler is not None:
                self.model_optimizer_scheduler.step()
 
            total_rsl = total_acc_cal(train_total_preds, train_total_labels)
            self.logger.info("     TrainLoss: {:.5f}\tTrainSupLoss: {:.5f}\tTrainCeLoss: {:.5f}\tTrainNum: {}\tAccTrainNum: {}\tTrainAcc: {:.5f}".format(
                np.mean(train_total_loss), np.mean(train_supcon_loss), np.mean(train_ce_loss), total_rsl["total_num"], total_rsl["correct_num"], total_rsl["accuracy"]))
            
            total_eval_rsl = self.eval_ce(phase="val", display=False)
            if (total_eval_rsl["accuracy"]) > best_acc:
                best_epoch = epoch
                best_acc = total_eval_rsl["accuracy"]
                best_model_weights['FEAT_MODEL'] = copy.deepcopy(self.networks['FEAT_MODEL'].state_dict())
                best_model_weights["PROJECTION_HEAD"] = copy.deepcopy(self.networks["PROJECTION_HEAD"].state_dict())
                best_model_weights['CLASSIFIER'] = copy.deepcopy(self.networks['CLASSIFIER'].state_dict())
                
        self.logger.info('===> Training Complete')
        self.logger.info('===> Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch))
        
        self.logger.info('===> Saving best checkpoint')
        model_file_path = self.save_model(epoch, best_epoch, best_model_weights, best_acc)
        self.logger.info('\t--> Best checkpoint is saved at {}'.format(model_file_path))
        
        self.reset_model(best_model_weights)
        self.eval_ce(phase="val", display=False)
        self.eval_ce(phase="test", display=False)
    

    def eval_ce(self, phase="val", display=False):
        if display:
            self.logger.info("-------------------- ****** Model {} ****** --------------------".format(phase))
        
        torch.cuda.empty_cache()
        for model in self.networks.values():
            model.eval()
        
        eval_total_loss = []
        eval_total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        eval_total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        for data, target, _ in self.dataloader_dict[phase]:
            data, target = data.float().to(self.device), target.long().to(self.device)

            with torch.set_grad_enabled(False):
                feature = self.networks["FEAT_MODEL"](data)
                logit = self.networks["CLASSIFIER"](feature)
                
                output = F.softmax(logit, 1)
                _, preds = output.max(dim=1)
                eval_total_preds = torch.cat((eval_total_preds, preds))
                eval_total_labels = torch.cat((eval_total_labels, target))

                loss = self.criterions["CE_LOSS"](logit, target)
                eval_total_loss.append(loss.item())
        
        total_rsl = total_acc_cal(eval_total_preds, eval_total_labels)
        
        if (phase == "val"):
                self.logger.info("     ValLoss: {:.5f}\tValNum: {}\tAccValNum: {}\t\tValAcc: {:.5f}".format(
                    np.mean(eval_total_loss), total_rsl["total_num"], total_rsl["correct_num"], total_rsl["accuracy"]))
        
        else:
            self.logger.info('===> Performance on test set')
            per_cls_rsl = each_cls_acc_cal(eval_total_preds, eval_total_labels)
            
            self.logger.info("     TestLoss: {:.5f}\tTestNum: {}\tAccTestNum: {}\tTestAcc: {:.5f}".format(
                np.mean(eval_total_loss), total_rsl["total_num"], total_rsl["correct_num"], total_rsl["accuracy"]))
            
            self.logger.info("     Per class accuracy: {}".format(per_cls_rsl["class_accs"]))
            self.logger.info("     Per class acc_Nums: {}".format(per_cls_rsl["per_cls_correct"]))
            self.logger.info("     Per class totalNum: {}".format(per_cls_rsl["per_cls_num"]))

        return total_rsl
