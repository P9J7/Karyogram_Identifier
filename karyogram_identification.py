import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import arch_models
from people_dataset import PeopleDataset
from utils import *
import heapq

def main():
    args = vars(parse_args_util())
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args['name'] = 'name of trained chromosome classifier'
    with open('models/%s/config.yml' % args['name'], 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)
    cudnn.benchmark = True
    print("=> Initial Model %s" % config['arch'])
    model = arch_models.__dict__[config['arch']](config['num_classes'])
    model = model.cuda()
    print('# Params of Model:', count_params(model))
    pth = torch.load('models/%s/models.pth' % config['name'])
    new_state_dict = OrderedDict()
    for k, v in pth.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    test_transform = Compose([
        transforms.Resize(224, 224),
        transforms.Normalize(),
        ToTensorV2()
    ])

    test_dataset = PeopleDataset(
        img_dir='path of test chromosome images which saved by peopleId',
        img_ext=config['img_ext'],
        transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('peopleId', []),
        ('label', []),
        ('predict', [])
    ])
    errorlog = OrderedDict([
        ('peopleId', []),
        ('class', []),
        ('predict', [])
    ])
    errorPeople_log = OrderedDict([
        ('peopleId', []),
        ('label', []),
        ('predict', [])
    ])
    labels = ['unreliable', 'correct']
    confusion = ConfusionMatrix(num_classes=2, labels=labels)
    with torch.no_grad():
        for imgs_people, classes_people, label, people_id in tqdm(test_loader, total=len(test_loader)):
            target = classes_people.cuda()
            imgs_people = torch.squeeze(imgs_people)
            input = imgs_people.cuda()
            output = model(input)
            prob = torch.softmax(output, 1)
            topk_prob, topk_class = torch.topk(prob, 5, 1)
            pred = topk_class[:, 0]
            pred_max = topk_class[:, 0]
            counter = torch.bincount(pred_max).cpu().numpy().tolist()
            min_index = []
            min_index_class = []
            for class_index, count in enumerate(counter):
                if count > 2 or ((len(counter) > 23) and (count > 1) and (class_index in [22, 23])):
                    exceed_index = [j for j, k in enumerate(pred_max) if k == class_index]
                    exceed_prob = [topk_prob[j, 0] for j in exceed_index]
                    if class_index in [22, 23]:
                        ex_num = count - 1
                    else:
                        ex_num = count - 2
                    min_prob = heapq.nsmallest(ex_num, exceed_prob)
                    for pr in min_prob:
                        index = exceed_prob.index(pr)
                        min_index.append(exceed_index[index])
                        min_index_class.append(class_index)
                        exceed_prob[index] = 0
            if len(min_index) > 0:
                apply_class_index = []
                if len(counter) == 24:
                    for j, k in enumerate(counter):
                        if k == 0 and j < 22:
                            apply_class_index.append(j)
                            apply_class_index.append(j)
                        elif k == 1 and j < 22:
                            apply_class_index.append(j)
                    if counter[22] == 0:
                        apply_class_index.append(22)
                elif len(counter) == 23:
                    for j, k in enumerate(counter):
                        if k == 0 and j < 23:
                            apply_class_index.append(j)
                            apply_class_index.append(j)
                        elif k == 1 and j < 23:
                            apply_class_index.append(j)
                    if len(min_index) != len(apply_class_index):
                        apply_class_index.append(23)
                        counter.append(0)
                for i in apply_class_index:
                    apply_class_prob = [prob[j, i] for j in min_index]
                    max_prob_index = apply_class_prob.index(max(apply_class_prob))
                    pred[min_index[max_prob_index]] = i
                    counter[i] = counter[i] + 1
                    counter[min_index_class[max_prob_index]] = counter[min_index_class[max_prob_index]] - 1
                    min_index.pop(max_prob_index)
                    min_index_class.pop(max_prob_index)
            accuracy = pred.eq(target.view_as(pred)).sum().item()
            if accuracy == len(imgs_people):
                people_predict = 1
            else:
                people_predict = 0
            confusion.update([people_predict], [label])
            if label != people_predict:
                errorPeople_log['peopleId'].append(people_id.cpu().item())
                errorPeople_log['label'].append(label.cpu().item())
                errorPeople_log['predict'].append(people_predict)
                for i, pred_class in enumerate(pred):
                    if pred_class != target[0][i]:
                        errorlog['peopleId'].append(people_id.cpu().item())
                        errorlog['class'].append(target[0][i].cpu().item()+1)
                        errorlog['predict'].append(pred_class.cpu().item()+1)
            log['peopleId'].append(people_id.cpu().item())
            log['label'].append(label.cpu().item())
            log['predict'].append(people_predict)
    confusion.plot()
    confusion.summary()
    pd.DataFrame(log).to_csv('models/%s/peopleId_label_pred.csv' % config['name'], mode='w', index=False)
    pd.DataFrame(errorlog).to_csv('models/%s/errorPeopleId_class_pred.csv' % config['name'], mode='w', index=False)
    pd.DataFrame(errorPeople_log).to_csv('models/%s/errorPeopleId_label_pred.csv' % config['name'], mode='w', index=False)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
