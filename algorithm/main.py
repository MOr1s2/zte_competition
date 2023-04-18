import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import myDataset
from model import MLP

parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', default='train')
args = parser.parse_args()

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

if args.mode == 'train':
    train_path = r'/home/liuchang/project/zte/data/train.csv'
    model_path = r'/home/liuchang/project/zte/model/cifar_net.pth'

    train_set = myDataset(train_path)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)

    net = MLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(20):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.to(torch.float32)
            labels = labels.long()

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:
                print(f'[{epoch + 1}, {i + 1:3d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0

    print('Training Done')

    torch.save(net.state_dict(), model_path)
    print('Nework storing Done')

else:
    test_path = r'/home/liuchang/project/zte/data/test.csv'
    model_path = r'/home/liuchang/project/zte/model/cifar_net.pth'
    result_path = r'/home/liuchang/project/zte/data/submit.json'

    test_set = myDataset(test_path, 'test')
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    net = MLP().to(device)
    net.load_state_dict(torch.load(model_path))

    result = {}

    with torch.no_grad():
        for data in test_loader:
            
            inputs = data[0].to(device)
            inputs = inputs.to(torch.float32)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            index = -1
            for p in predicted:
                index += 1
                result.update({str(index): p.item()})
    print('Test Done')
    count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
    for value in result.values():
        count[value] += 1
    print(f'result: {count}')
    with open(result_path, 'w') as f:
        f.write(json.dumps(result))
    print('Write Done')