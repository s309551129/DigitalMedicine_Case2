import argparse
import csv
import torch
import torch.nn as nn
import torch.utils.data as data
from models import WSDAN
from torch.autograd import Variable
from utils import batch_augment
from evalLoader import Dataset


def get_info(csv_path):
    csv_file = open(csv_path, newline='')
    rows = csv.reader(csv_file, delimiter=',')
    seq = []
    for row in rows:
        if row[0] == "FileID": continue
        seq.append(row[0])
    
    return seq


def write_ans(preds, filenames, fout):
    preds = torch.argmax(preds, dim=1)
    for i in range(len(preds)):
        if preds[i].item() == 0:
            fout.write(filenames[i]+",Negative"+"\n")
        elif preds[i].item() == 1:
            fout.write(filenames[i]+",Typical"+"\n")
        else:
            fout.write(filenames[i]+",Atypical"+"\n")


def eval():
    model.eval()
    fout = open("./submission.csv", 'w')
    fout.write("FileID,Type\n")
    with torch.no_grad():
        for step, (imgs, filenames) in enumerate(dataLoader):
            imgs = Variable(imgs).to(args.device)
            # Raw Image
            pred_raw, _,  _, attention_map = model(imgs)
            # Object Localization and Refinement
            crop_images = batch_augment(imgs, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
            pred_crop, _, _, _ = model(crop_images)
            # Final prediction
            preds = (pred_raw + pred_crop) / 2.
            write_ans(preds, filenames, fout)


def set_model():
    model = WSDAN(M=args.num_attentions, net=args.net, pretrained=True)
    model = nn.DataParallel(model).to(args.device)
    model.load_state_dict(torch.load('./weight/resnet34'))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="None")
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_attentions', default=32, type=int)
    parser.add_argument('--net', default='resnet34', type=str)
    parser.add_argument('--img_size', default=224, type=int)
    args = parser.parse_args()
    print('Single Model Inference')

    seq = get_info("./data/sample_submission.csv")
    dataset = Dataset(root="./data/testing/", seq=seq, img_size=args.img_size)
    dataLoader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = set_model()
    print("Number of testing set: {}".format(len(dataset)))
    eval()
