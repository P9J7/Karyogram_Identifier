import os
import arch_models
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import losses
from png_dataset import PNGDataset
from utils import *
from torch.utils.tensorboard import SummaryWriter
ARCH_NAMES = arch_models.__all__
LOSS_NAMES = losses.__all__


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None,
                        help='models name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNest50',
                        choices=ARCH_NAMES,
                        help='models architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: ResNet50)')
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=24, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=224, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=224, type=int,
                        help='image height')
    parser.add_argument('--loss', default='LabelSmoothing',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: LabelSmoothing)')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: SGD)')
    parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--Tmax', default=64, type=int, help='CosineAnnealingLR T_max')
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--resume', default=False, type=str2bool, help='resume models train')
    parser.add_argument('--resume_epoch', default=-1, type=int, help='resume epoch')
    parser.add_argument('--save_freq', default=1, type=int, help='pth save frequency')
    parser.add_argument('--dataset_path', default='path of extracted chromosome images for training', help='train dataset path')
    config = parser.parse_args()
    return config


def set_random_seed(seed=10, deterministic=True, benchmark=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True


def train(train_loader, model, criterion, optimizer, accumulation_steps=1):
    avg_meters = {'loss': AverageMeter(),
                  'accuracy': AverageMeter()
                  }
    model.train()
    pbar = tqdm(total=len(train_loader))
    for i, (imgs, classes, _) in enumerate(train_loader):
        accuracy = 0
        imgs = imgs.cuda()
        classes = classes.cuda()
        output = model(imgs)
        loss = criterion(torch.squeeze(output), classes)
        loss = loss / accumulation_steps
        pred = output.max(1, keepdim=True)[1]
        accuracy += pred.eq(classes.view_as(pred)).sum().item()
        loss.backward()
        if ((i+1) % accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()
        avg_meters['loss'].update(loss.item(), imgs.size(0))
        avg_meters['accuracy'].update(accuracy/imgs.size(0), imgs.size(0))
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('accuracy', avg_meters['accuracy'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('accuracy', avg_meters['accuracy'].avg)])


def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'accuracy': AverageMeter()}
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for imgs, classes, _ in val_loader:
            accuracy = 0
            imgs = imgs.cuda()
            classes = classes.cuda()
            output = model(imgs)
            loss = criterion(torch.squeeze(output), classes)
            pred = output.max(1, keepdim=True)[1]
            accuracy += pred.eq(classes.view_as(pred)).sum().item()
            avg_meters['loss'].update(loss.item(), imgs.size(0))
            avg_meters['accuracy'].update(accuracy/imgs.size(0), imgs.size(0))
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('accuracy', avg_meters['accuracy'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('accuracy', avg_meters['accuracy'].avg)])


def main():
    set_random_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = vars(parse_args())
    config['arch'] = 'name of model'

    os.makedirs('models/%s' % config['name'], exist_ok=True)
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)
    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)
    criterion = losses.__dict__[config['loss']]().cuda()
    print("=> Initial Model %s" % config['arch'])
    model = arch_models.__dict__[config['arch']](config['num_classes'])
    model = model.cuda()
    model = nn.DataParallel(model)

    print('# Params of Model:', count_params(model))
    writer = SummaryWriter(config['name'] + '_log')
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['Tmax'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    if config['resume']:
        path_checkpoint = 'models/{}/ckpt_{}.pth'.format(config['name'], config['resume_epoch'])
        checkpoint = torch.load(path_checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        model.load_state_dict(checkpoint['model_state_dict'])
        log = load_resume_log('models/{}/log.csv'.format(config['name']))
        best_acc = float(max(log['val_accuracy']))
    else:
        start_epoch = 0
        best_acc = 0
        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('train_loss', []),
            ('train_accuracy', []),
            ('val_loss', []),
            ('val_accuracy', []),
            ('is_best_epoch', [])
        ])

    data = get_stratify(config['dataset_path'])
    train_img_ids, val_img_ids, _, _ = train_test_split(data, data['class'], test_size=0.2, train_size=0.8, random_state=41, stratify=data['class'])
    train_transform = Compose([
        transforms.Flip(0.5),
        transforms.ElasticTransform(p=0.3),
        transforms.OpticalDistortion(p=0.2),
        transforms.ShiftScaleRotate(rotate_limit=15, p=0.1),
        transforms.Normalize(),
        ToTensorV2()
    ])

    val_transform = Compose([
        transforms.Resize(224, 224),
        transforms.Normalize(),
        ToTensorV2()
    ])

    train_dataset = PNGDataset(
        imgs_df=train_img_ids,
        img_ext=config['img_ext'],
        transform=train_transform
    )
    val_dataset = PNGDataset(
        imgs_df=val_img_ids,
        img_ext=config['img_ext'],
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=True)

    trigger = 0
    for epoch in range(start_epoch + 1, config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        train_log = train(train_loader, model, criterion, optimizer, 4)
        val_log = validate(val_loader, model, criterion)
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
        print('lr %.4f - loss %.4f - accuracy %.4f - val_loss %.4f - val_accuracy %.4f'
              % (optimizer.state_dict()['param_groups'][0]['lr'], train_log['loss'], train_log['accuracy'], val_log['loss'], val_log['accuracy']))
        writer.add_scalar('val/loss', val_log['loss'], epoch)
        writer.add_scalar('val/acc', val_log['accuracy'], epoch)
        writer.add_scalar('train/acc', train_log['accuracy'], epoch)
        writer.add_scalar('train/loss', train_log['loss'], epoch)
        writer.add_scalar('train/lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        log['epoch'].append(epoch)
        log['lr'].append(optimizer.state_dict()['param_groups'][0]['lr'])
        log['train_loss'].append(train_log['loss'])
        log['train_accuracy'].append(train_log['accuracy'])
        log['val_loss'].append(val_log['loss'])
        log['val_accuracy'].append(val_log['accuracy'])

        trigger += 1

        if epoch % config['save_freq'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict()
            }
            torch.save(checkpoint, 'models/{}/ckpt_{}.pth'.format(config['name'], epoch))

        if val_log['accuracy'] > best_acc:
            log['is_best_epoch'].append('yes')
            torch.save(model.state_dict(), 'models/%s/models.pth' %
                       config['name'])
            best_acc = val_log['accuracy']
            print("=> saved best models in epoch{}".format(epoch))
            trigger = 0
        else:
            log['is_best_epoch'].append('no')
        pd.DataFrame(log).to_csv('models/%s/log.csv' % config['name'], mode='w', index=False)
        if (config['early_stopping'] >= 0) and (trigger >= config['early_stopping']):
            print("=> early stop of model")
            break
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
