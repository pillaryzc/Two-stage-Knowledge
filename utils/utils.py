import torch
import os
from colorama import Style, Fore, Back
import numpy as np
import importlib
from . import pytorch_ssim
from skimage.metrics import structural_similarity as sk_cpt_ssim

"""
### 模块名称: Model Performance Evaluation and Management Utility

#### 功能简介
- **主要作用**: 提供了一套工具和函数，用于在图像恢复和增强任务中评估模型性能（如PSNR和SSIM），并根据性能指标管理模型的保存。
- **关键功能**:
  - `save_top_k`函数用于在训练过程中保存表现最好的k个模型。
  - `torchPSNR`和`cal_psnr`函数计算预测图像与真实图像之间的峰值信噪比(PSNR)。
  - `cal_ssim`函数评估预测图像与真实图像的结构相似性(SSIM)，支持不同的天气类型对SSIM计算方法的选择。

#### 关键代码段解析
- **save_top_k**:
  - **功能**: 根据PSNR值管理和保存模型的最佳状态。
  - **重要代码解释**: 当新模型的性能超过已保存模型中的最低性能时，会替换该模型，并且可能会删除性能最差的模型。

- **torchPSNR & cal_psnr**:
  - **功能**: 计算并返回预测图像和真实图像之间的PSNR值。
  - **重要代码解释**: `torchPSNR`使用PyTorch实现PSNR的计算，`cal_psnr`根据天气类型选择合适的PSNR计算方法。

- **cal_ssim**:
  - **功能**: 根据天气类型选择合适的方法计算预测图像与真实图像之间的SSIM值。
  - **重要代码解释**: 对于包含雨天的图像，使用PyTorch实现的SSIM；否则，使用`skimage`库计算SSIM。

#### 输入输出
- **输入**: 
  - 预测图像和真实图像张量，以及其他函数特定的参数（如模型状态、优化器状态、调度器状态等）。
- **输出**: 
  - PSNR和SSIM计算结果，或者是更新后的模型保存状态。

#### 依赖关系
- **内部依赖**: 依赖于PyTorch、NumPy、Colorama、`skimage.metrics`以及自定义的`pytorch_ssim`模块。

#### 注意事项
- **特别注意**: 在使用`cal_psnr`和`cal_ssim`函数时，确保输入图像的维度和类型符合预期，特别是在不同天气类型下的SSIM计算中。

#### 通俗的形象举例
想象你正在举办一场摄影比赛，参赛者提交了多张照片。你需要根据照片的清晰度和细节（PSNR和SSIM）来评分。`save_top_k`就像是你只保留得分最高的几张照片，`torchPSNR`和`cal_ssim`则是用来计算每张照片的得分。不同的计算方法（根据天气类型）就像是在不同的光照和天气条件下评估照片的质量。

#### 如何使用
```python
# 假设pred_image和gt_image是预测和真实图像的PyTorch张量
weather_type = 'sunny'  # 或 'rain' 根据实际情况

# 计算PSNR
psnr_value = cal_psnr(pred_image, gt_image, weather_type)
print(f"PSNR: {psnr_value}")

# 计算SSIM
ssim_value = cal_ssim(pred_image, gt_image, weather_type)
print(f"SSIM: {ssim_value}")

# 假设有模型、优化器、调度器以及保存目录
model, optimizer, scheduler = None, None, None  # 示例，需替换为实际对象
save_dir = "./model_saves"
top_k_state = [] 

 # 维护一个列表，保存表现最好的模型状态
epoch = 1  # 当前训练轮次

# 根据模型性能保存最佳模型
top_k_state = save_top_k(model, optimizer, scheduler, top_k_state, k=3, epoch=epoch, save_dir=save_dir, psnr=psnr_value, ssim=ssim_value)
```
这段代码展示了如何在实际的图像恢复或增强任务中使用这些工具来评估模型性能，并根据PSNR和SSIM值管理模型的保存。

"""

def get_func(path):
	module = path[:path.rfind('.')]
	model_name = path[path.rfind('.') + 1:]
	mod = importlib.import_module(module)
	net_func = getattr(mod, model_name)

	return net_func


def save_top_k(model, optimizer, scheduler, top_k_state, k, epoch, save_dir, psnr, ssim):
	flag = False
	popped_state = {}
	model_path = os.path.join(save_dir, 'epoch_{}_psnr{:.3f}_ssim{:.3f}'.format(epoch, psnr, ssim))

	if len(top_k_state) < k or psnr >= top_k_state[-1]['psnr']:
		
		if len(top_k_state) >= k:
			popped_state = top_k_state.pop()
			os.remove(os.path.join(save_dir, 'epoch_{}_psnr{:.3f}_ssim{:.3f}'.format(popped_state['epoch'], popped_state['psnr'], popped_state['ssim'])))

		flag = True
		top_k_state.append({'epoch': epoch, 'psnr': psnr, 'ssim': ssim})
		scheduler = scheduler.state_dict() if scheduler is not None else None
		torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler}, model_path)

	top_k_state.sort(key = lambda s: s['psnr'], reverse = True)
	if flag:
		if popped_state == {}:
			print(Back.RED + 'PSNR: {:.3f} , length of buffer < {}'.format(psnr, k))
		else:
			print(Back.RED + 'PSNR: {:.3f}  >=  last PSNR: {:.3f}'.format(psnr, popped_state['psnr']))
		print('Save the better model!!!' + Style.RESET_ALL)
		print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)
	else:
		print(Back.GREEN + 'PSNR: {:.3f}  <  rank-3 PSNR: {:.3f}'.format(psnr, top_k_state[-1]['psnr']))
		print('Do not save this model, QQQ' + Style.RESET_ALL)
		print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

	return top_k_state


@torch.no_grad()
def torchPSNR(prd_img, tar_img):
	if not isinstance(prd_img, torch.Tensor):
		prd_img = torch.from_numpy(prd_img)
		tar_img = torch.from_numpy(tar_img)

	imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
	rmse = (imdff**2).mean().sqrt()
	ps = 20 * torch.log10(1/rmse)
	return ps


def rgb2ycbcr(img, only_y=True):
	'''same as matlab rgb2ycbcr
	only_y: only return Y channel
	Input:
		uint8, [0, 255]
		float, [0, 1]
	'''
	if isinstance(img, torch.Tensor):
		img = img.clamp(0., 1.)
		img = img.cpu().detach().permute(1, 2, 0).numpy()

	in_img_type = img.dtype
	img.astype(np.float32)
	if in_img_type != np.uint8:
		img *= 255.
	
	# convert
	if only_y:
		rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
	else:
		rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
							  [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
	if in_img_type == np.uint8:
		rlt = rlt.round()
	else:
		rlt /= 255.
	if rlt.ndim != 3:
		rlt = np.expand_dims(rlt, axis=-1)
		
	return rlt.astype(in_img_type)


def cal_psnr(pred_image, gt_image, weather_type):
	# shape of pred_image and gt_image: [1, 3, H, W]
	if 'rain' in weather_type:
		return torchPSNR(rgb2ycbcr(pred_image[0]), rgb2ycbcr(gt_image[0]))
	else:
		return torchPSNR(pred_image, gt_image)


def cal_ssim(pred_image, gt_image, weather_type):
	# shape of pred_image and gt_image: [1, 3, H, W]
	if 'rain' in weather_type:
		return pytorch_ssim.ssim(pred_image, gt_image).item()
	else:
		return sk_cpt_ssim(rgb2ycbcr(pred_image[0]), rgb2ycbcr(gt_image[0]), data_range=1.0, multichannel=True).item()
