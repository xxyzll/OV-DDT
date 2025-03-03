from mmengine.hooks import ProfilerHook
from mmyolo.registry import HOOKS
from typing import Dict, Optional, Sequence, Union
from mmengine.hooks import Hook
import os, json

@HOOKS.register_module()
class SaveResultHook(Hook):
    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch,
                        outputs: Optional[Sequence] = None) -> None:
        save_dir = runner.work_dir
        current_result = {}
        for data_sample, output in zip(data_batch['data_samples'], outputs):
            image_id = data_sample['metainfo']['img_id']
            current_result[image_id] = {
                'bboxes': output.pred_instances.bboxes.tolist(),
                'labels': output.pred_instances.labels.tolist(),
                'scores': output.pred_instances.scores.tolist(),
                'decisions': output.pred_instances.decisions.tolist()
            }
        json_file_path = os.path.join(save_dir, 'results.json')
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                existing_results = json.load(f)
        else:
            existing_results = {}
        existing_results.update(current_result)
        
        with open(json_file_path, 'w') as f:
            json.dump(existing_results, f, indent=4)