import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from parser import get_args
from datasets import get_dataloader
from utils import get_model
def main():
    args = get_args()

    shape = (32, 32, 3)

    # Define Dataloader
    train_loader, valid_loader, test_loader = get_dataloader(args)
    
    # Define Model
    model = get_model(args, shape)

    if torch.cuda.device_count() >= 1:
        print(f'Model in to {torch.cuda.device_count()}, with type : {torch.cuda.get_device_name()}')
        model = model.cuda()

    # Define loss function + optimizer + scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = lrs.MultiStepLR(optimizer, milestones=[10, 25, 50], gamma=0.1)

    # Define SWA model, scheduler
    swa_model = optim.swa_utils.AveragedModel(model)
    swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr=args.swa_lr)

    # Train
    

    for e in range(args.epochs):
        model.train()
        total_loss = 0
        count = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.cuda(), labels.cuda()

            outputs = model(data)
            loss = criterion(outputs, labels)
            
            if len(labels.size()) > 1:
                labels = torch.argmax(labels, axis=1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.tolist()
            count += labels.size(0)

            if batch_idx % args.log_interval == 0:
                print(f'Epoch:{e:0>3}[{batch_idx}/{len(train_loader)}]--Train loss:{total_loss / count:.4f}')
            
            scheduler.step()

        if e > args.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

    print('Training done')
    
    # Evaluate
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            data, labels = data.cuda(), labels.cuda()
            
            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f'Test accuracy : {100 * correct / total}')




if __name__ == '__main__':
    main()