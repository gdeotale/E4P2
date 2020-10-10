import torch
from tqdm import tqdm
from torch.autograd import Variable

def train(netG, netD, batch_size, device, trainloader, optimizerG, optimizerD, criterion, epoch):
  netG.train()
  netD.train()
  pbar = tqdm(trainloader)
  running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
  running_loss = 0.0
  for i, (data1, target1) in enumerate(pbar):
        # get the inputs
        data = data1[0]
        target = data1[1]
        data, target = data.to(device), target.to(device)
        running_results['batch_sizes'] += batch_size
        
        # zero the parameter gradient
        real_img = Variable(data)
        z = Variable(target)
        fake_img = netG(z)
        
        netD.zero_grad()
        real_out = netD(real_img).mean()
        fake_out = netD(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        netG.zero_grad()
        fake_img = netG(z)
        fake_out = netD(fake_img).mean()
        g_loss = criterion(fake_out, fake_img, real_img)
        g_loss.backward()
        
        optimizerG.step()

        # Predict
        # loss for current batch before optimization 
        running_results['g_loss'] += g_loss.item() * batch_size
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += real_out.item() * batch_size
        running_results['g_score'] += fake_out.item() * batch_size
    
        pbar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, 25, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
  return running_results, netG, netD, optimizerG, optimizerD