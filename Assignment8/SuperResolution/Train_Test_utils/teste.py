import torch
from tqdm import tqdm
import pytorch_ssim
from math import log10

def test(netG, device, testloader, criterion, batch_size):
    netG.eval()
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    pbar1 = tqdm(testloader)
    with torch.no_grad():
        for i, (data, target) in enumerate(pbar1): 
          valing_results['batch_sizes'] += batch_size
          val_lr = data[0].to(device)
          hr = data[1].to(device)
          val_hr_restore = data[2].to(device)
          sr = netG(val_lr)
          batch_mse = ((sr - hr) ** 2).data.mean()
          valing_results['mse'] += batch_mse * batch_size
          batch_ssim = pytorch_ssim.ssim(sr, hr).item()
          valing_results['ssims'] += batch_ssim * batch_size
          valing_results['psnr'] = 10 * log10(((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes'])))
          valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
          pbar1.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
    return valing_results