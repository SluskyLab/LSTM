# LSTM
#
# Copyright (C) 2026 University of Kansas
#
# Daniel Montezano
# University of Kansas
# Slusky Lab

#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#

# lstm_lm_bidirectional.py
#
# Basic Long Short-Term Memory network for TMBB generation.
# It has code both for training and for inference which should be selected with
# argument --task.
#
# The model is the one-layer bidirectional LSTM.
#
# NOTE: The scheduler is set-up with patience=5 and factor=0.5.
# NOTE: The scheduler can be disabled (uses constant lr) with --no_scheduler flag.
# NOTE: Different methods of generation can be used.
# NOTE: There is a function to detect stickiness when nucleus 0.6 is used.


import re
import glob
import time
import random
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

# define the model class using torch
class TMBBLanguageModel(nn.Module):
	def __init__(self,
				input_dim,
				hidden_dim,
				output_dim):
		super(TMBBLanguageModel, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.lstm = nn.LSTM(
							input_size=self.input_dim,
							hidden_size=self.hidden_dim,
							num_layers=1, bidirectional=True,
							bias=True)
		self.fc1 = nn.Linear(
							in_features=self.hidden_dim*2,
							out_features=self.output_dim,
							bias=True)
	def forward(self, X):
		lstm_out, last_state = self.lstm(X)
		ss = last_state[0].split(1,dim=0)
		ss = torch.cat(ss,dim=2)
		residue_space = self.fc1(ss)
		residue_scores = F.log_softmax(residue_space, dim=2)
		return residue_scores

	def greedy_sampling(self, logits):
		selected_index = torch.argmax(logits)
		return selected_index

	def pure_sampling(self, logits, temperature=1.0):
		logits = logits / temperature
		selected_index = torch.multinomial( torch.exp(logits) / torch.exp(logits).sum() , 1)
		return selected_index

	def topk_sampling(self, logits, k): # from cyrilzakka llm playbook
		assert type(k) == int
		top_k_indices = torch.argsort(logits)[-k:]
		top_k_logits = logits[top_k_indices]
		top_k_probs = torch.exp(top_k_logits) / torch.sum(torch.exp(top_k_logits))
		m = torch.distributions.Categorical(top_k_probs)
		selected_index = top_k_indices[m.sample()]
		return selected_index

	def nucleus_sampling(self, logits, p): # from cyrilzakka llm playbook
		sorted_indices = torch.argsort(logits)
		sorted_probs = torch.exp(logits[sorted_indices]) / torch.sum(torch.exp(logits))
		cum_probs = torch.cumsum(sorted_probs, dim=0)
		valid_indices = torch.where(cum_probs >= (1 - p))[0]
		if len(valid_indices) > 0:
			min_valid_index = valid_indices[0]
			mask = sorted_indices[min_valid_index:]
			selected_probs = logits[mask].exp() / logits[mask].exp().sum()
			selected_index = mask[torch.multinomial(selected_probs,1)]
		else:
			mask = sorted_indices[-1]
			selected_index = mask
		return selected_index

	def is_sticky(self, sequenz):
		# disregard sequences with less than two residues (we count the EOL character)
		if len(sequenz) <= 2:
			return {'flag': True, 'score': 100.0 }# return as if it was a repetition. It will prevent generation of empty seqs
		# define the list of valid amino acids
		aa_list = "ARNDCQEGHILKMFPSTWYV"
		aa_count = {key: 0 for key in aa_list}
		# we are including the impact of kmers from 2 to 100 residues in length
		kmer = range(2,101) # <-- needs to go past one from desired maximum kmer
		# a dict indexed by amino acids with a list for each kmer count as value initialized at zero
		kmer_count = {key: [0]*len(kmer) for key in aa_list}
		# keep track of the counts of each amino acid type
		for key in aa_list:
			aa_count[key] += sequenz.count(key)
		# loop over each of the residues in the sequence
		count = 1
		curpos = 1 # we need to start at the second residue, which is okay because we are testing length of sequenz
		while curpos < len(sequenz):
			if sequenz[curpos] == sequenz[curpos-1]:
				count += 1
			else:
				if count > 1: # if we have any kmer we need to update our counts
					addition = list(map(lambda x : count//x, list(range(2,count+1))))
					for k in range(min(count-1,99)): # we are simply discarding k-mers longer than 100 residues
						kmer_count[sequenz[curpos-1]][k] += addition[k]
					count = 1 # reset the counter
			curpos += 1
		# finally, in case there was a k-mer at the very end of the sequence
		if count > 1:
			addition = list(map(lambda x : count//x, list(range(2,count+1))))
			for k in range(min(count-1,99)): # we are simply discarding k-mers longer than 100 residues
				kmer_count[sequenz[curpos-1]][k] += addition[k]
		# this current code counts - for each kmer size - non-overlapping kmers. However, it does count overlapping
		# kmers if they are of different size.
		# also check the presence of long KEIs, GASTVLs, PAs
		additional_re_score = 0
		pattern_GAVLST = "[G|A|V|L|S|T]{30,}"
		pattern_KEI = "[K|E|I]{30,}"
		pattern_PA = "[P|A]{30,}"
		matches = re.findall(pattern_GAVLST, sequenz)
		matches_lenks = list(map(len, matches))
		additional_re_score += sum([a*b for a,b in Counter(matches_lenks).items()])
		matches = re.findall(pattern_KEI, sequenz)
		matches_lenks = list(map(len, matches))
		additional_re_score += sum([a*b for a,b in Counter(matches_lenks).items()])
		matches = re.findall(pattern_PA, sequenz)
		matches_lenks = list(map(len, matches))
		additional_re_score += sum([a*b for a,b in Counter(matches_lenks).items()])
		# compute the scores
		score = additional_re_score
		for aa in aa_list:
			if aa_count[aa] > 0:
				score += sum([a*b for a,b in zip(kmer,kmer_count[aa])])
		score = score / len(sequenz)
		# the following threshold of 0.2 is taken from the KDE of stickiness scores of the
		# iiab_tmbb training and validation sets. It provides a stringent cut-off.
		if score >= 0.2:
			return {'flag': True, 'score': score} # the sequence is repetitive
		else:
			return {'flag': False, 'score': score}  # the sequence looks good

class TMBBDataset(torch.utils.data.Dataset):
	'''INPUT ARG dataset: train | valid'''
	def __init__(self, basename, alphabet, frag_size, batch_size, dataset):
		self.basename = basename
		self.alphabet = alphabet
		self.frag_size = frag_size
		self.batch_size = batch_size
		self.dataset = dataset
		self.restart_point = 0
		fh = open(self.basename + '_' + self.dataset + '.fasta')
		self.seqs = fh.readlines()
		fh.close()
		self.start_token_stretch = self.alphabet[0]  * self.frag_size
		self.end_token_stretch   = self.alphabet[21] * self.frag_size
		# add start and end tokens; filter our FASTA headers; make as list
		self.seqs = [self.start_token_stretch + xs.strip() + self.end_token_stretch for xs in self.seqs if not '>' in xs]
		# create the indexing table for fragments
		self.seqs_lenks = [len(k) for k in self.seqs]
		self.num_seqs = len(self.seqs)
		# table lists cumulative number of possible fragments
		self.frag_table = []
		current_count = 0 # how many possible fragments so far?
		for k in range(self.num_seqs):
			self.last_resi = self.seqs_lenks[k] - self.frag_size
			self.frag_table.append(current_count + self.last_resi)
			current_count += self.last_resi
		print("We have",self.frag_table[-1],"fragments")

	def __len__(self):
		# give the last cumulative number
		return self.frag_table[-1]

	def __getitem__(self, idx):
		# search through the list of cumsum numbers find
		# the seq ID containing this global residue ID
		seqi = self.restart_point
		cumsum = 0
		while idx >= self.frag_table[seqi]:
			cumsum = self.frag_table[seqi]
			seqi = seqi + 1
		self.restart_point = max(0,seqi - 1)
		# the residue number is computed relative to the sequence
		res_beg = idx - cumsum
		res_end = res_beg + self.frag_size + 1
		# convert the fragments to a numerical index
		# translate chars to number indices
		fragment = list(self.seqs[seqi][res_beg:res_end])
		target = fragment.pop()
		fragment = [char_to_int[x] for x in fragment]
		target = char_to_int[target]
		encoded_fragment = torch.nn.functional.one_hot(torch.as_tensor(fragment),len(self.alphabet))
		sample = {"encoded_fragment":torch.as_tensor(encoded_fragment,dtype=torch.float32),"target":torch.as_tensor(target)} # make sure fragment are floats so that they can be multiplied by model weights.
		return sample

class SimpleCustomBatch:
	def __init__(self,data):
		encoded_fragment = [item['encoded_fragment'] for item in data]
		target = [item['target'] for item in data]
		self.inp = torch.stack(encoded_fragment,dim=1)
		self.tgt = torch.stack(target)
	def pin_memory(self):
		self.inp = self.inp.pin_memory()
		self.tgt = self.tgt.pin_memory()
		return {"encoded_fragment":self.inp, "target":self.tgt}

def collate_restructure(data):
	return SimpleCustomBatch(data)


############### TRAINING FUNCTION
def execute_training():
	################# INSTANTIATE DATASET
	train_dataset = TMBBDataset(args.basename, alphabet, args.lstm_frag_size, args.batch_size, 'train')
	valid_dataset = TMBBDataset(args.basename, alphabet, args.lstm_frag_size, args.batch_size, 'valid')
	################# INSTANTIATE DATALOADERS
	train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=False,pin_memory=True,num_workers=4,drop_last=True,collate_fn=collate_restructure)
	valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=False,pin_memory=True,num_workers=4,drop_last=True,collate_fn=collate_restructure)
	################## MODEL INSTANTIATION
	lstmnet = TMBBLanguageModel(len(alphabet),args.lstm_num_nodes,len(alphabet))
	lstmnet.to(device)
	################## OPTIMIZER INSTANTIATION
	optim_dict = {
	'sgd': torch.optim.SGD(lstmnet.parameters(),lr=args.lr,momentum=0.8),
	'adam': torch.optim.Adam(lstmnet.parameters(),lr=args.lr),
	'rmsprop': torch.optim.RMSprop(lstmnet.parameters(),lr=args.lr)
	}
	optimizer = optim_dict[args.optimizer]
	################## PLATEAU SCHEDULER INSTANTIATION
	if not args.no_scheduler:
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
	################## LOSS FUNCTION DEFINITION
	loss_fn = torch.nn.NLLLoss(reduction='mean') # 'mean' is already the default.
	################## TRAINING LOOP
	evofile = open(args.outfolder + "/train_evolution.tsv",'w')
	evofile.write("EPOCH\tTRAINLOSS\tTRAINACCU\tVALIDLOSS\tVALIDACCU\tTOPNACCU\n")
	evofile.close()
	epoch = 0
	torch.save({"epoch":epoch,
				"model_state_dict":lstmnet.state_dict(),
				"optimizer_state_dict":optimizer.state_dict()}, args.outfolder + "/model_" + str(epoch))
	for epoch in range(1,args.num_epochs+1):
		t0 = time.time()
		train_dataset.restart_point = 0
		valid_dataset.restart_point = 0
		train_loss = 0.0
		train_positives = 0
		for i,b in enumerate(train_dataloader):
			X = b['encoded_fragment']
			y = b['target']
			X = X.to(device)
			y = y.to(device)
			lstmnet.zero_grad()

			y_pred = lstmnet(X).squeeze()
			loss = loss_fn(y_pred, y)
			loss.backward()
			optimizer.step()
			# accumulate the loss from every batch
			train_loss += loss.detach()
			# accumulate the argmax training accuracy
			train_positives += y_pred.argmax(dim=1).eq(y).sum()
		t1 = time.time()
		print("training:   EPOCH {:02d} TOOK {:.4f} SECS".format(epoch,t1-t0),flush=True)

		lstmnet.eval() # this takes care of dropout and batch normalization layers
		with torch.no_grad(): # this prevents gradient computation and saves processing
			t0 = time.time()
			valid_loss = 0.0
			valid_positives = 0
			valid_topk_positives = 0
			k = 5
			for i,b in enumerate(valid_dataloader):
				X = b['encoded_fragment']
				y = b['target']
				X = X.to(device)
				y = y.to(device)
				y_pred = lstmnet(X).squeeze()
				loss = loss_fn(y_pred, y)
				# accumulate the loss from every batch
				valid_loss += loss.detach()
				# accumulate the argmax validation accuracy
				valid_positives += y_pred.argmax(dim=1).eq(y).sum()
				# accumulate the top-k validation accuracy
				valid_topk_positives += ((y_pred.topk(k=k,dim=1)[1]).eq(y.view([-1,1]))).sum()
			evofile = open(args.outfolder + "/train_evolution.tsv",'a')
			evofile.write(str(epoch))
			evofile.write('\t' + "{:.4f}".format(train_loss.item()*args.batch_size/len(train_dataset)))
			evofile.write('\t' + "{:.4f}".format(train_positives.item()/len(train_dataset)*100.0))
			evofile.write('\t' + "{:.4f}".format(valid_loss.item()*args.batch_size/len(valid_dataset)))
			evofile.write('\t' + "{:.4f}".format(valid_positives.item()/len(valid_dataset)*100.0))
			evofile.write('\t' + "{:.4f}".format(valid_topk_positives.item()/len(valid_dataset)*100.0))
			evofile.write('\n')
			evofile.close()
			t1 = time.time()
			print("validation: EPOCH {:02d} TOOK {:.4f} SECS".format(epoch,t1-t0),flush=True)
		# run one step of LR scheduler and save model for this epoch (can be used for training or for resuming training)
		if not args.no_scheduler:
			scheduler.step(valid_loss)
		torch.save({"epoch":epoch,
					"model_state_dict":lstmnet.state_dict(),
					"optimizer_state_dict":optimizer.state_dict()}, args.outfolder + "/model_" + str(epoch))
		lstmnet.train()
# END OF execute_training()

############### INFERENCE FUNCTION
def execute_inference():
	# if all we want is to sample a single epoch, just go for it
	if args.single_epoch:
		start_epoch = args.num_epochs
	# otherwise we need to check what is available
	else:
		if glob.glob(args.outfolder + '/epoch_0.fasta') == []: # we have not done any inference yet. Start from zero.
			start_epoch = 0
		else:
			last_available_epoch = max([int(re.sub('.*epoch_','',re.sub('.fasta','',ff))) for ff in glob.glob(args.outfolder + '/epoch_*')])
			fh = open(args.outfolder + "/valid_evolution.tsv")
			for line in fh.readlines():
				pass # we only want the last line
			fh.close()
			last_available_valid = int(line.split('\t')[0])
			if last_available_epoch != last_available_valid:
				print("ERROR: There is a mismatch between the last epoch saved and the last validation data available.")
				exit(1)
			else:
				start_epoch = last_available_epoch + 1
		print("inference: START_EPOCH is {}".format(start_epoch),flush=True)
		print("inference: END_EPOCH is {} (inclusive)".format(args.num_epochs),flush=True)
	################# INSTANTIATE DATASET
	valid_dataset = TMBBDataset(args.basename, alphabet, args.lstm_frag_size, args.batch_size, 'valid')
	################# INSTANTIATE DATALOADERS
	valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=False,pin_memory=True,num_workers=4,drop_last=True,collate_fn=collate_restructure)
	################## MODEL INSTANTIATION
	lstmnet = TMBBLanguageModel(len(alphabet),args.lstm_num_nodes,len(alphabet))
	lstmnet.to(device)
	if start_epoch == 0: # be sure not to overwrite the file if it already exists
		evofile = open(args.outfolder + "/valid_evolution.tsv",'w')
		evofile.write("EPOCH\tPPL\tAVGENTROPY\tPERCENTG\n")
		evofile.close()
	with torch.no_grad(): # this prevents gradient computation and saves processing
		################# execute the sampling for each epoch
		for epoch in range(start_epoch,args.num_epochs+1): # note the loop limits to include all epochs the user requested.
			t0 = time.time()
			valid_dataset.restart_point = 0
			################ LOAD PRE-TRAINED MODEL
			checkpoint = torch.load(args.outfolder + "/model_" + str(epoch))
			lstmnet.load_state_dict(checkpoint['model_state_dict'])
			assert epoch == checkpoint['epoch']
			lstmnet.eval() # this takes care of dropout and batch normalization layers
			################ SAMPLE A BATCH OF SEQUENCES
			outfile = open(args.outfolder + "/epoch_" + str(epoch) + ".fasta",'w')
			max_sequence_len = 600
			number_generated = 0
			while number_generated < args.sample_size:
				seed = ['^'] * args.lstm_frag_size
				generated_sequence = ''
				for i in range(max_sequence_len):
					seed_int = torch.as_tensor([char_to_int[x] for x in seed])
					seed_enc = torch.as_tensor(torch.nn.functional.one_hot(seed_int,len(alphabet)),dtype=torch.float32).view([args.lstm_frag_size,-1,len(alphabet)])
					seed_enc = seed_enc.to(device)
					logits = lstmnet(seed_enc).flatten()
					# sample one residue using different methods
					if args.method == "greedy":
						next_residue_idx = lstmnet.greedy_sampling(logits)
					elif args.method == "pure":
						next_residue_idx = lstmnet.pure_sampling(logits, temperature=args.tparam)
					elif args.method == "topk":
						next_residue_idx = lstmnet.topk_sampling(logits, k=args.kparam)
					elif args.method == "nucleus":
						next_residue_idx = lstmnet.nucleus_sampling(logits, p=args.pparam)
					next_residue_chr = alphabet[next_residue_idx]
					if next_residue_idx == 21 or next_residue_idx == 0:
						break
					else:
						generated_sequence += next_residue_chr
						seed = seed[1:] + [next_residue_chr]
				stickiness = lstmnet.is_sticky(generated_sequence)
				number_generated += 1
				if stickiness['flag'] and args.sticky_filter:
					pass # we want to generate again
				else:
					outfile.write(">PROTEIN_" + str(number_generated) + " is_sticky_at_0.2: " + str(stickiness['flag']) + " stickiness_score: " + str(stickiness['score']) + '\n')
					outfile.write(generated_sequence + '\n')
					number_generated += 1
			outfile.close()
			t1 = time.time()
			print("sampling: EPOCH {:02d} TOOK {:.4f} SECS".format(epoch, t1-t0), flush=True)

			################### COMPUTE PERFORMANCE METRICS
			t0 = time.time()
			entropy = 0.0
			perplexity = 0.0
			gly_in_topk = 0
			gly_idx = alphabet.index('G')
			k = 3
			for i,b in enumerate(valid_dataloader):
				##### PREDICT ON THE VALIDATION DATA
				X = b['encoded_fragment']
				y = b['target']
				X = X.to(device)
				y = y.to(device)
				y_pred = lstmnet(X).squeeze()
				##### ACCUMULATE VALUES FOR ENTROPY OF DISTRIBUTIONS
				entropy += -1.0 * (y_pred.exp() * y_pred).sum()
				##### ACCUMULATE VALUES FOR PERPLEXITY
				perplexity += y_pred.gather(-1,y.view([-1,1])).sum()
				##### ACCUMULATE TIMES GLYCINE IN TOP-K
				gly_in_topk += (y_pred.topk(k=k,dim=1)[1]).eq(gly_idx).sum()
			evofile = open(args.outfolder + "/valid_evolution.tsv",'a')
			evofile.write(str(epoch))
			evofile.write('\t' + "{:.4f}".format(torch.exp(-1.0*perplexity/len(valid_dataset)).item()))
			evofile.write('\t' + "{:.4f}".format(entropy.item()/len(valid_dataset)))
			evofile.write('\t' + "{:.4f}".format(gly_in_topk.item()/len(valid_dataset)*100.0))
			evofile.write('\n')
			evofile.close()
			t1 = time.time()
			print("metrics:  EPOCH {:02d} TOOK {:.4f} SECS".format(epoch,t1-t0),flush=True)
# END OF execute_inference()

def execute_restart():
	last_available_model = max([int(re.sub('.*model_','',ff)) for ff in glob.glob(args.outfolder + '/model_*')])
	if args.num_epochs <= last_available_model:
		print("Nothing to do. We have all the models already.")
		exit(0)
	fh = open(args.outfolder + "/train_evolution.tsv")
	for line in fh.readlines():
		pass # we only want the last line
	fh.close()
	last_available_epoch = int(line.split('\t')[0])
	if last_available_epoch != last_available_model:
		print("ERROR: There is a mismatch between the last model saved and the last training data available.")
		exit(1)
	print("restart: LAST AVAILABLE MODEL is {}".format(last_available_model),flush=True)
	print("restart: START_EPOCH is {}".format(last_available_epoch+1),flush=True)
	print("restart: END_EPOCH is {} (inclusive)".format(args.num_epochs),flush=True)
	################# INSTANTIATE DATASET
	train_dataset = TMBBDataset(args.basename, alphabet, args.lstm_frag_size, args.batch_size, 'train')
	valid_dataset = TMBBDataset(args.basename, alphabet, args.lstm_frag_size, args.batch_size, 'valid')
	################# INSTANTIATE DATALOADERS
	train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=False,pin_memory=True,num_workers=4,drop_last=True,collate_fn=collate_restructure)
	valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=False,pin_memory=True,num_workers=4,drop_last=True,collate_fn=collate_restructure)
	################## MODEL INSTANTIATION
	lstmnet = TMBBLanguageModel(len(alphabet),args.lstm_num_nodes,len(alphabet))
	lstmnet.to(device)
	################## OPTIMIZER INSTANTIATION
	optim_dict = {
	'sgd': torch.optim.SGD(lstmnet.parameters(),lr=args.lr,momentum=0.8),
	'adam': torch.optim.Adam(lstmnet.parameters(),lr=args.lr),
	'rmsprop': torch.optim.RMSprop(lstmnet.parameters(),lr=args.lr)
	}
	optimizer = optim_dict[args.optimizer]
	################## PLATEAU SCHEDULER INSTANTIATION
	if not args.no_scheduler:
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
	################## LOSS FUNCTION DEFINITION
	loss_fn = torch.nn.NLLLoss(reduction='mean') # 'mean' is already the default.
	################ LOAD PRE-TRAINED MODEL FROM LAST AVAILABLE EPOCH
	checkpoint = torch.load(args.outfolder + "/model_" + str(last_available_epoch))
	lstmnet.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	assert last_available_epoch == checkpoint['epoch']
	lstmnet.train()
	################## TRAINING LOOP
	for epoch in range(last_available_epoch+1,args.num_epochs+1): # note the loop limits to ensure model is saved after the number of epochs of training
		t0 = time.time()
		train_dataset.restart_point = 0
		valid_dataset.restart_point = 0
		train_loss = 0.0
		train_positives = 0
		for i,b in enumerate(train_dataloader):
			X = b['encoded_fragment']
			y = b['target']
			X = X.to(device)
			y = y.to(device)
			lstmnet.zero_grad()
			y_pred = lstmnet(X).squeeze()
			loss = loss_fn(y_pred, y)
			loss.backward()
			optimizer.step()
			# accumulate the loss from every batch
			train_loss += loss.detach()
			# accumulate the argmax training accuracy
			train_positives += y_pred.argmax(dim=1).eq(y).sum()
		t1 = time.time()
		print("training:   EPOCH {:02d} TOOK {:.4f} SECS".format(epoch,t1-t0),flush=True)

		lstmnet.eval() # this takes care of dropout and batch normalization layers
		valid_loss = 0.0
		valid_positives = 0
		valid_topk_positives = 0
		k = 5
		with torch.no_grad(): # this prevents gradient computation and saves processing
			t0 = time.time()
			for i,b in enumerate(valid_dataloader):
				X = b['encoded_fragment']
				y = b['target']
				X = X.to(device)
				y = y.to(device)
				y_pred = lstmnet(X).squeeze()
				loss = loss_fn(y_pred, y)
				# accumulate the loss from every batch
				valid_loss += loss.detach()
				# accumulate the argmax validation accuracy
				valid_positives += y_pred.argmax(dim=1).eq(y).sum()
				# accumulate the top-k validation accuracy
				valid_topk_positives += ((y_pred.topk(k=k,dim=1)[1]).eq(y.view([-1,1]))).sum()
			evofile = open(args.outfolder + "/train_evolution.tsv",'a') # this file has already been created. Do not ovewrite!
			evofile.write(str(epoch))
			evofile.write('\t' + "{:.4f}".format(train_loss.item()*args.batch_size/len(train_dataset)))
			evofile.write('\t' + "{:.4f}".format(train_positives.item()/len(train_dataset)*100.0))
			evofile.write('\t' + "{:.4f}".format(valid_loss.item()*args.batch_size/len(valid_dataset)))
			evofile.write('\t' + "{:.4f}".format(valid_positives.item()/len(valid_dataset)*100.0))
			evofile.write('\t' + "{:.4f}".format(valid_topk_positives.item()/len(valid_dataset)*100.0))
			evofile.write('\n')
			evofile.close()
			t1 = time.time()
			print("validation: EPOCH {:02d} TOOK {:.4f} SECS".format(epoch,t1-t0),flush=True)
		# run one step of LR scheduler and save model for this epoch (to be used for training or for further resuming training)
		if not args.no_scheduler:
			scheduler.step(valid_loss)
		torch.save({"epoch":epoch,
					"model_state_dict":lstmnet.state_dict(),
					"optimizer_state_dict":optimizer.state_dict()}, args.outfolder + "/model_" + str(epoch))
		lstmnet.train()
# END OF execute_restart()

############### START MAIN PROGRAM
if __name__ == '__main__':
	print("Python version:     " + sys.version,flush=True)
	print("NumPy version:      " + np.__version__,flush=True)
	print("PyTorch version:    " + torch.__version__,flush=True)
	parser = argparse.ArgumentParser("PROTLSTM")
	parser.add_argument("basename",help="name of the train/valid sets.")
	parser.add_argument("outfolder",help="folder where to save output.")
	parser.add_argument("num_epochs",type=int,help="number of epochs.")
	parser.add_argument("sample_interval",type=int,help="sample interval.")
	parser.add_argument("batch_size",type=int,help="size of minibatch.")
	parser.add_argument("optimizer",help="optimizer (sgd|adam|rmsprop).")
	parser.add_argument("lr",type=float,help="initial learning rate.")
	parser.add_argument("sample_size",type=int,help="size to samples.")
	parser.add_argument("--lstm_num_nodes",type=int,help="state size.")
	parser.add_argument("--lstm_frag_size",type=int,help="fragment size.")
	parser.add_argument("--task",help="Task to perform (training | inference | restart)")
	parser.add_argument("--gpu_device",help="The GPU number to use ( 0 | 1 )")
	parser.add_argument("--no_scheduler",action="store_true",help="Disables use of scheduler and makes learning rate constant.")
	parser.add_argument("--sticky_filter",action="store_true",help="Enables the stickiness filter.")
	parser.add_argument("--single_epoch",action="store_true",help="Enables inference of only one single epoch.")
	parser.add_argument("--method", help="Generation method (greedy | pure | topk | nucleus | beam)")
	parser.add_argument("--tparam", type=float, help="The temperature parameter for pure generation")
	parser.add_argument("--kparam", type=int, help="The k parameter for top-k generation")
	parser.add_argument("--pparam", type=float, help="The p parameter for top-p generation")
	parser.add_argument("--beam_width", type=int, help="The number of beams for beam search")
	args = parser.parse_args()
	if parser.prog == "PROTLSTM":
		if not args.lstm_num_nodes:
			print("For this LSTM model you need to specify")
			print("--lstm_num_nodes.")
			exit()
		if not args.lstm_frag_size:
			print("For this LSTM model you need to specify")
			print("--lstm_frag_size.")
			exit()
		if args.task=="inference" and not args.method:
			print("For inference you need to specify the method of generation")
			print("--method.")
			exit()
		if args.method=="topk" and not args.kparam:
			print("For top-k generation you need to specify K")
			print("--kparam.")
			exit()
		if args.method=="nucleus" and not args.pparam:
			print("For top-p generation you need to specify P")
			print("--pparam.")
			exit()
		if args.method=="pure" and not args.tparam:
			print("For pure generation you need to specify temperature")
			print("--tparam.")
			exit()

	################# SET GLOBAL TORCH OPTIONS
	torch.set_printoptions(linewidth=200)
	################# INITIALIZE RANDOM NUMBER GENERATOR
	#rng = torch.random.manual_seed(45)
	#rng_init_state = rng.get_state()
	#torch.random.set_rng_state(rng_init_state)
	#torch.manual_seed(23)
	#random.seed(11)
	#np.random.seed(24)
	################# SETUP GPU
	if torch.cuda.is_available():
		print("CUDA is available.",flush=True)
		print("CUDA current device:",torch.cuda.current_device(),flush=True)
		print("CUDA device count:",torch.cuda.device_count(),flush=True)
		print("CUDA device name:",torch.cuda.get_device_name(0),flush=True)
		device = "cuda:" + str(args.gpu_device)
	else:
		print("CUDA is not present.",flush=True)
		device = "cpu"
	################# DEFINE ALPHABET
	alphabet = ['^','A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','Z']
	################# DEFINE CONVERSION FUNCTION
	char_to_int = { k:v for v,k in enumerate(alphabet) }
	################# EXECUTE SELECTED TASK
	if args.task == 'training':
		execute_training()
	if args.task == 'inference':
		execute_inference()
	if args.task == 'restart':
		execute_restart()

#560
