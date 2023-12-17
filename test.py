import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def test(data_loader, net, device):
    net.eval()
    print("=========Testing Network==========")
    with torch.no_grad():
        for index, (input, target) in enumerate(data_loader):
            
            input = torch.tensor(input, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)
            output = net(input)
            
            prediction = torch.max(output, 0)
            correct = torch.max(target, 0)
            
            print("--test " + str(index+1) + "--")
            print("certainty:", str(prediction[0].item()*100)+"%")
            print("prediction:", prediction[1].item())
            print("target:", correct[1].item())
            
            if prediction[1].item() == correct[1].item():
                print('\x1b[6;30;42m' + 'Success!' + '\x1b[0m')
            else:
                print('\x1b[6;30;41m' + 'Fail' + '\x1b[0m')