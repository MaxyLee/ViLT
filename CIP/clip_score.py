from numpy.lib.function_base import iterable
import torch
import clip
import re
from PIL import Image
import torchvision
from torchvision import transforms

class CLIPScore:
    def __init__(self, model="ViT-B/32", device='cpu'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.tensor2image = torchvision.transforms.ToPILImage()
        print(f'Load CLIPScore("{model}") to "{device}"')
    
    def single_image_preprocess(self, img):
        example_type = type(img)
        if example_type == str:
            img = self.preprocess(Image.open(img)).unsqueeze(0).to(self.device)
        elif example_type == torch.Tensor:
            img = self.preprocess(self.tensor2image(img)).unsqueeze(0).to(self.device)
        else:
            img = self.preprocess(img).unsqueeze(0).to(self.device)
        return img

    @torch.no_grad()
    def score(self, texts, images, return_diag=False):
        if isinstance(texts, str):
            texts = [texts]
        texts = [re.sub('\s##|\[CLS\]|\[SEP\]', '', text).strip() for text in texts]
        texts = clip.tokenize(texts).to(self.device)

        if isinstance(images, (list, tuple)):
            images = torch.cat([self.single_image_preprocess(img) for img in images], dim=0)
        else:
            images = self.single_image_preprocess(images)

        logits_per_image, logits_per_text = self.model(images, texts)
        if return_diag:
            bsz = logits_per_image.shape[0]
            logits = logits_per_image[torch.arange(bsz), torch.arange(bsz)]
        else:
            logits = logits_per_image
        
        return logits

if __name__ == '__main__':
    captions = [
        'A person standing on top of a ski covered slope.',
        'A person on skis and with poles in the snow and facing the blue sky.',
        'A person standing on skiis on the snowy slope.',
        'A skier stands on skis at the top of a snowy plateau.',
        'A person is skiing on a snowy hill top.',
        'This is a father of two children.'
    ]

    image_fns = ['/data2/share/data/coco/images/val2017/000000002532.jpg']
    image = [Image.open(image_fn) for image_fn in image_fns]
    image_tensor = [transforms.ToTensor()(Image.open(image_fn)) for image_fn in image_fns]

    clip_score = CLIPScore()
    score_fn = clip_score.score(captions, image_fns, return_diag=False)
    score_image = clip_score.score(captions, image, return_diag=False)
    score_tensor = clip_score.score(captions, image, return_diag=False)

    import ipdb; ipdb.set_trace()
        
        
        
        