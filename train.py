import numpy as np 
import torch
import argparse
import logging 
import os 
import time
import json  
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

from torchvision import dataset, transforms
from torchvision import models



# Data Loading from inbuilt cifar dataset 

train_loader = torch.utils.data.DataLoader(dataset.CIFAR10(root = './', train = True, download = True, transform = transforms.Compose([
                                                        transform.RandomCrop(32, padding = 4),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        # transform.Normalize ?
                                                        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])))

test_loader = torch.utils.data.DataLoader(dataset.CIFAR10(root = './', train = False, download = True, transform = transforms.Compose([
                                                        transforms.ToTensor(),
                                                        # transform.Normalize ?
                                                        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])))



# Normal Training of the network 

def train(model, optimizer, loss_fn, dataloader, metrics, params):

    # set the model training as active
    model.train()
    summ = []
    

    with tqdm(total = len(dataset)) as t :
        for i, (train_batch, label_batch) in enumerate(dataloader):

            # move to gpu if present 
            if params.cuda:
                train_batch, label_batch = train_batch.cuda(async = True), label_batch.cuda(async = True)

            # Convert to torch variable Variable has data, grad field
            train_batch, label_batch = Variable(train_batch), Variable(label_batch)

            output_batch = model(train_batch)
            loss = loss_fn(output_batch, label_batch)

            # clear previous gradients, compute gradients of all variables wrt loss

            optimizer.zero_grad()
            loss.backward()

            # next step of updates

            optimizer.step()

            if i % params.save_summary_steps == 0:

                # conert the output to cpu , convert to numpy 

                output_batch = output_batch.data.cpu().numpy()
                labels_batch = label_batch.data.cpu().numpy()

                summary_batch = { metric:metrics[metric](output_batch,label_batch) for metric in metrics } 
                summary_batch['loss'] = loss.data[0]

                summ.append(summary_batch)

            loss_avg.update(loss.data[0])

            t.set_postfix(loss = '{:05.3f}'.format(loss_avg()))
            t.update()

        metrics_mean = { metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = ";".join("{}:{:05.3f}".format(k,v) for k,v in metrics_mean.items())
        logging.info(" Train metrics: "+ metrics_string)

def fetch_teacher_outputs(teacher_model, dataloader, params):
    
    # Set the teacher model to evaluation 
    teacher_model.eval()
    teacher_outputs = []

    for i, (data_batch, label_batch) in enumerate(dataloader):
        if params.cuda:
            data_batch, label_batch = data_batch.cuda(async = True), label_batch.cuda(async = True)
        
        data_batch, label_batch = Variable(data_batch), Variable(label_batch)

        output_teacher = teacher_model(data_batch).data.cpu().numpy()
        teacher_outputs.append(output_teacher)
    return teacher_outputs

def train_kd(model, optimizer, teacher_outputs, loss_fn_kd, dataloader, metrics, params):

    model.train()

    summ = []
    loss_avg = utils.RunningAverage()

    with tqdm(total = len(dataloader)) as t:
        for i, (train_batch,label_batch) in enumerate(dataloader):
            if params.cuda:
                train_batch, label_batch = train_batch.cuda(async = True), label_batch(async = True)

            train_batch, label_batch = Variable(train_batch), Variable(label_batch)

            output_batch = model(train_batch)

            output_teacher = torch.from_numpy(teacher_outputs[i])

            loss = loss_fn_kd(output_batch, label_batch, teacher_outputs, params)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if i % params.save_summary_steps == 0:

                output_batch = output_batch.data.cpu().numpy()
                label_batch = label_batch.data.cpu().numpy()

                summary_batch = { metric:metrics[metric](output_batch,label_batch) for metric in metrics } 
                summary_batch['loss'] = loss.data[0]

                summ.append(summary_batch)

            loss_avg.update(loss.data[0])

            t.set_postfix(loss = '{:05.3f}'.format(loss_avg()))
            t.update()

        metrics_mean = { metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = ";".join("{}:{:05.3f}".format(k,v) for k,v in metrics_mean.items())
        logging.info(" Train metrics: "+ metrics_string)



if __name__ == '__main__':

    json_path = ''
    assert os.path.isfile(json_path), "No json found at {} ".format(json_path)
    with open(json_path) as f:
        params = json.load(f)
    
    param.cuda = torch.cuda.is_available()

    model = net.VGG(params).cuda() if params.cuda else net.VGG(params)
    optimizer = optim.Adam(model.parameters(), lr = params.learning_rate)
    loss_fn_kd = net.loss_function_kd
    metrics = net.metrics

    teacher_model = resnet.resnet18().cuda() if params.cuda else resnet.resnet18()
    
    teacher_checkpoint = ''

    if not os.path.exists(teacher_checkpoint):
        raise("File not present at {}".format(teacher_checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location = lambda storage, loc : storage)

    teacher_model.load_state_dict(checkpoint['state_dict'])
    # teacher_ouputs = 
    # train_kd()








            


