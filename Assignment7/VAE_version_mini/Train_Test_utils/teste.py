import torch
from tqdm import tqdm

def test(net, device, testloader, criterion):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar1 = tqdm(testloader)
        for i, (data, target) in enumerate(pbar1):
           data, target = data.to(device), target.to(device)
           outputs = net(data)
           _, predicted = torch.max(outputs.data, 1)
           loss = criterion(outputs, target)
           total += target.size(0)
           correct += (predicted == target).sum().item()
        print("Test Loss=", loss.cpu().numpy(), "Test Accuracy=",100*correct/total)
        #pbar1.update(1)  
        #print('Loss %0.2f Accuracy of the network on the 10000 test images: %0.2f', loss, (100 * correct / total))
    test_acc = (100 * correct / total)  

    return test_acc, loss.cpu().numpy()

def test_categorywise(net, device, testloader, classes):
   class_correct = list(0. for i in range(10))
   class_total = list(0. for i in range(10))
   with torch.no_grad():
      pbar = tqdm(testloader)
      for i, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        outputs = net(data)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == target).squeeze()
        for i in range(4):
            label = target[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


   for i in range(10):
      print('Accuracy of %5s : %0.2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))