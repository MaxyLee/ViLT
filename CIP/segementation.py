import torch
import pixellib

from pixellib.torchbackend.instance import instanceSegmentation

# model_path = '/data/private/mxy/projects/mmda/ckpt/pointrend/pointrend_resnet50.pkl'
# ins = instanceSegmentation()
# ins.load_model(model_path)
# # ins.segmentBatch("inputfolder", show_bboxes=True, extract_segmented_objects=True, extract_from_box=True,
# # save_extracted_objects=True, output_folder_name="outputfolder")
# ins.segmentImage("tmp/4817681157.jpg", show_bboxes=True, extract_segmented_objects=True,
# save_extracted_objects=True, output_image_name="tmp/output_image.jpg" )
model_pc = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB3_phrasecut', pretrained=False, return_postprocessor=False)
ckpt = torch.load('checkpoints/phrasecut_EB3_checkpoint.pth')
model_pc.load_state_dict(ckpt['model'])
model_pc = model_pc.cuda()
model_pc.eval()

import ipdb; ipdb.set_trace()