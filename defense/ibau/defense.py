import sys
sys.path.append("/home/Xuxiong/experiment")
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import hypergrad as hg
import ibau_cfg
from loader import Box
from evaluate import test

if __name__ == "__main__":
	opt = ibau_cfg.get_arguments().parse_args()
	opt.attack_type = "all-to-one"
	device = opt.device
	box = Box(opt)
	save_path = box.get_save_path()
	_, _, model = box.get_state_dict()
	tlabel = box.tlabel

	unlloader = box.get_dataloader(train="clean", batch_size=opt.batch_size, shuffle=True)
	cln_testloader = box.get_dataloader(train="test", batch_size=opt.batch_size, shuffle=False)

	### initialize theta
	criterion = nn.CrossEntropyLoss()
	outer_opt = torch.optim.Adam(model.parameters(), lr=opt.lr)

	### define the inner loss L2
	def loss_inner(perturb, model_params):
		images = images_list[0].to(device)
		labels = labels_list[0].long().to(device)
	#     per_img = torch.clamp(images+perturb[0],min=0,max=1)
		per_img = images+perturb[0]
		per_logits = model.forward(per_img)
		loss = F.cross_entropy(per_logits, labels, reduction='none')
		loss_regu = torch.mean(-loss) +0.001*torch.pow(torch.norm(perturb[0]),2)
		return loss_regu

	### define the outer loss L1
	def loss_outer(perturb, model_params):
		portion = 0.01
		images, labels = images_list[batchnum].to(device), labels_list[batchnum].long().to(device)
		patching = torch.zeros_like(images, device=device)
		number = images.shape[0]
		rand_idx = random.sample(list(np.arange(number)),int(number*portion))
		patching[rand_idx] = perturb[0]
	#     unlearn_imgs = torch.clamp(images+patching,min=0,max=1)
		unlearn_imgs = images+patching
		logits = model(unlearn_imgs)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(logits, labels)
		return loss

	images_list, labels_list = [], []
	for index, (images, _, _, labels) in enumerate(unlloader):
		images_list.append(images)
		labels_list.append(labels)
	inner_opt = hg.GradientDescent(loss_inner, 0.1)


	### inner loop and optimization by batch computing
	model.eval()
	print("Before Unlearning")
	print("Test BA:")
	ba = test(testloader=cln_testloader, testmodel=model, box=box, poisoned=False)
	print("Test ASR:")
	asr = test(testloader=cln_testloader, testmodel=model, box=box, poisoned=True, poitarget=True, passlabel=tlabel)

	for n in range(opt.n_rounds):
		batch_pert = torch.zeros([1, opt.size, opt.size], requires_grad=True, device=device)
		batch_opt = torch.optim.SGD(params=[batch_pert],lr=10)

		for images, _, _, labels in unlloader:
			images = images.to(device)
			ori_lab = torch.argmax(model.forward(images),axis = 1).long()
	#         per_logits = model.forward(torch.clamp(images+batch_pert,min=0,max=1))
			per_logits = model.forward(images+batch_pert)
			loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
			loss_regu = torch.mean(-loss) + 0.001*torch.pow(torch.norm(batch_pert),2)
			batch_opt.zero_grad()
			loss_regu.backward(retain_graph = True)
			batch_opt.step()
		#l2-ball
		# pert = batch_pert * min(1, 10 / torch.norm(batch_pert))
		pert = batch_pert
		#unlearn step         
		for batchnum in range(len(images_list)): 
			outer_opt.zero_grad()
			hg.fixed_point(pert, list(model.parameters()), opt.K, inner_opt, loss_outer) 
			outer_opt.step()
		print(f"After Unlearning Round {n}" )
		print("Test BA:")
		ba = test(testloader=cln_testloader, testmodel=model, box=box, poisoned=False)
		print("Test ASR:")
		asr = test(testloader=cln_testloader, testmodel=model, box=box, poisoned=True, poitarget=True, passlabel=tlabel)

		

