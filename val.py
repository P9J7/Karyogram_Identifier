import os
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import arch_models
from png_dataset import PNGDataset
from utils import *
from sklearn.model_selection import train_test_split


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    args = vars(parse_args_util())
    args['name'] = 'name of trained model'
    with open('models/%s/config.yml' % args['name'], 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)
    cudnn.benchmark = True
    print("=> Initial Model %s" % config['arch'])
    model = arch_models.__dict__[config['arch']](config['num_classes'])
    model = model.cuda()
    print('# Params of Model:', count_params(model))
    data_df = get_stratify('path of extracted chromosome images for training')
    train_img_ids, val_img_ids, _, _ = train_test_split(data_df, data_df['class'], test_size=0.2, train_size=0.8,
                                                        random_state=41, stratify=data_df['class'])
    pth = torch.load('models/%s/models.pth' % config['name'])
    new_state_dict = OrderedDict()
    for k, v in pth.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    val_transform = Compose([
        transforms.Resize(224, 224),
        transforms.Normalize(),
        ToTensorV2()
    ])

    val_dataset = PNGDataset(
        imgs_df=val_img_ids,
        img_ext=config['img_ext'],
        transform=val_transform
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('imgId', []),
        ('class', []),
        ('pred', [])
    ])

    labels = []
    for i in range(1, 23):
        labels.append(str(i))
    labels.append('x')
    labels.append('y')
    confusion = ConfusionMatrix(num_classes=24, labels=labels)
    with torch.no_grad():
        for imgs, classes, imgs_id in tqdm(val_loader, total=len(val_loader)):
            imgs = imgs.cuda()
            classes = classes.cuda()
            output = model(imgs)
            outputs = torch.softmax(output, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.cpu().numpy(), classes.cpu().numpy())
            pred = output.max(1, keepdim=True)[1]
            pred_cpu = pred.cpu().numpy().tolist()
            for i in range(len(imgs_id)):
                log['imgId'].append(imgs_id[i])
                log['class'].append(classes[i]+1)
                log['pred'].append(pred_cpu[i][0]+1)
    confusion.plot()
    confusion.summary()
    pd.DataFrame(log).to_csv('models/%s/imgId_class_pred.csv' %
                             config['name'], mode='w', index=False)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
