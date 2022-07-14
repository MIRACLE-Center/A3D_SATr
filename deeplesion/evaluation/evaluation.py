from mmdet.core.evaluation.eval_hooks import DistEvalHook
from deeplesion.evaluation.evaluation_metrics import sens_at_FP
import numpy as np
import torch.distributed as dist
from mmcv.parallel import collate, scatter
from mmcv.runner import Hook
from torch.utils.data import Dataset
from mmdet import datasets
import mmcv
import torch
import os
import os.path as osp


class MyDeepLesionEval(Hook):

    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset, {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        # if runner.rank == 0:
        #     prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=False, **data_gpu)
            results[idx] = (result[0], data_gpu['gt_bboxes'], data_gpu['thickness'])
            # results[idx] = (result[0], data_gpu['gt_bboxes'])#hanadd

            #batch_size = runner.world_size
            # if runner.rank == 0:
            #     for _ in range(batch_size):
            #         prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self, runner, results):
        gt_bboxes = []
        gt_labels = []
        pred_bboxes = []
        s1_box=[]
        s1_gt=[]
        s5_box=[]
        s5_gt=[]
        for i in range(len(results)):
            pred_bboxes.append(results[i][0][0])
            gt_bboxes.append(results[i][1][0].cpu().numpy())
            if results[i][2][0][0]<=2.:
                s1_box.append(pred_bboxes[-1])
                s1_gt.append(gt_bboxes[-1])
            if results[i][2][0][0] > 2.:
                s5_box.append(pred_bboxes[-1])
                s5_gt.append(gt_bboxes[-1]) 
        avgFP=[0.5, 1, 2, 4, 8, 16]
        iou_th_astrue=0.5
        # try:
        r = sens_at_FP(pred_bboxes, gt_bboxes, avgFP, iou_th_astrue)
        # except Exception as e:
        #     print('froc crash!\n',e)
        #     return
        r1 = sens_at_FP(s1_box, s1_gt, avgFP, iou_th_astrue)
        r2 = sens_at_FP(s5_box, s5_gt, avgFP, iou_th_astrue)
        # except Exception as e:
        #     print('froc crash!\n',e)
        #     return

        runner.logger.info(f"{r}")
        with open('./logs/log_all_metrics.txt','a') as f:
            f.writelines(f"{runner.epoch+1}:{r}:\t{runner.work_dir.split('/')[-1]}\t{runner.cfg.description}\t{r1}\t{r2}\n")