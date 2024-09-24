import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec

#This Class is a custom layer in PyTorch that performs parametric whitening. 
class PWLayer(nn.Module): # In this case, "parametric" indicates that some parameters are learned during training
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__() #initializer of the parent class (nn.Module) 

        self.dropout = nn.Dropout(p=dropout) #this will randomly zeroes some of the elements of the input tensor. 
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True) #Initi.a bias parameter for the layer with zeros
        self.lin = nn.Linear(input_size, output_size, bias=False) # linear transformation from input to output_size

        self.apply(self._init_weights) #Applies the _init_weights method to all sub-modules of PWLayer

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)#Ensures that the weights start with small random values,
                                                            # which can help the model learn effectively

    def forward(self, x): #forward function
        return self.lin(self.dropout(x) - self.bias) 
    #Applies dropout to x, randomly zeroing some of the elements and subtracts the learned bias from the input.

class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    #method initializes the MoEAdaptorLayer class, which implements a (MoE) mechanism with parametric whitening, noisy gating.
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        #Calls the initializer of the parent class (nn.Module) to ensure that the module is properly initialized.
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps #no of expert layers in MOE mechanism. Each expert is a neural network layer. 
        self.noisy_gating = noise #A boolean flag indicating whether to use noisy gating during training.
                                             #input size, output size
        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)]) #list of expert layers
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True) #compute gating score
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True) #add noise to gating score


    #applies a gating mechanism to decide which experts to use in a Mixture of Experts (MoE) model.    
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2): #x: input tensor, train: a boolean integer for training and evaluation
        clean_logits = x @ self.w_gate #performs dot product between 'x' & gating weights result is tensor 
        if self.noisy_gating and train: #add noise in the training mode
            raw_noise_stddev = x @ self.w_noise #calculate noise standard deviation 
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon)) # fn to ensure +ve values & small constant added
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev) #Generate Noise 
            logits = noisy_logits
        else:
            logits = clean_logits #if model is not in training mode i.e. Train is 'False'

        gates = F.softmax(logits, dim=-1) #Apply softmax to commute Gates
        return gates 

    #the forward method is responsible for applying the 'Mixture of Experts (MoE)' mechanism to the input 'x'
    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) #(B, n_E) #calls the method noisy_top to compute the gating scores.
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)] #output of each expert
        expert_outputs = torch.cat(expert_outputs, dim=-2) #concatenates the outputs of all experts along the newly added dim.
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs #Apply Gating Scores to Expert Outputs of shape:(B, n_E, D)
        return multiple_outputs.sum(dim=-2) #sum across the expert dimensions (n_E) resulting tensor (B,D) a final output


class UniSRec(SASRec):
    def __init__(self, config, dataset): #constructor method that initializes the instance of the class & takes 2 parameters
        super().__init__(config, dataset) #calls the constructor of the SASRec class to initialize the base part of UniSRec
        #Configuration parameters:
        self.train_stage = config['train_stage'] #determines the mode in which model wil be trained
        self.temperature = config['temperature'] #stores a temp. parameter which will be used in softmax technique 
        self.lam = config['lambda'] #stroes a 'lambda' parameter used as a regularization coefficient
        #validation of the training Stage:
        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'  #This ensures that the training stage is correctly specified.
        #Conditional Initialization:    
        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None
            # for `transductive_ft`, `item_embedding` is defined in SASRec base model
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            # `plm_embedding` in pre-train stage will be carried via dataloader
            self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
        # MoE Adaptor Initialization:
        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob']
        )
    #responsible for processing the input data through the model's layers and generating the final output:
    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device) #creates a tensor of sequential indices 
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq) #Reshapes & expands to match the dim. of item_seq tensor
        position_embedding = self.position_embedding(position_ids) #This adds info. about the position of each item in sequence.
        
        #Combine Item Embeddings with Position Embeddings:
        input_emb = item_emb + position_embedding #Add both types of info. to create a richer rep. of each item
        if self.train_stage == 'transductive_ft':
            input_emb = input_emb + self.item_embedding(item_seq) #adds additional item embedding for if cond.
        input_emb = self.LayerNorm(input_emb) #Applies Layer Normalization to input_emb
        input_emb = self.dropout(input_emb) #Applies dropout to input_emb to prevent overfitting
        
        #model focus on relevant parts of the sequence and ignore padded or irrelevant sections
        extended_attention_mask = self.get_attention_mask(item_seq)

        #Pass Through Transformer Encoder i.e. Processes the sequence to generate meaningful representations.
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H] #B= Batch size, H = Hidden dimension 



    #calculate the contrastive loss b/w the output sequence embeddings and the embeddings of +ive and -ive items
    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction): #3 parameters
        pos_items_emb = self.moe_adaptor(interaction['pos_item_emb']) #Obtain the embeddings of positive items 
        pos_items_emb = F.normalize(pos_items_emb, dim=1) #and normalize them
        #Compute the logits for the positive items
        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)
        #Compute the logits for the negative items    
        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)
        #Calculate the contrastive loss
        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    #is a method designed to calculate a contrastive loss between sequences of embeddings useful for training models
    def seq_seq_contrastive_task(self, seq_output, same_pos_id, interaction):
                    # embeddings of the original sequences, A boolean tensor, A dictionary containing additional info-
        
        #Retrieves and processes the augmented sequences and their embeddings
        item_seq_aug = interaction[self.ITEM_SEQ + '_aug']
        item_seq_len_aug = interaction[self.ITEM_SEQ_LEN + '_aug']
        item_emb_list_aug = self.moe_adaptor(interaction['item_emb_list_aug'])
        #Forward Pass for Augmented Sequences:  
        seq_output_aug = self.forward(item_seq_aug, item_emb_list_aug, item_seq_len_aug)
        seq_output_aug = F.normalize(seq_output_aug, dim=1)
        #Computes the logits for positive pairs
        pos_logits = (seq_output * seq_output_aug).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)
        #Computes the logits for negative pairs
        neg_logits = torch.matmul(seq_output, seq_output_aug.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)
        #Calculates the contrastive loss.
        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    #the goal of this function is to train the model by leveraging contrastive learning
    def pretrain(self, interaction):
        
        #Extracting Inputs
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(interaction['item_emb_list'])
        #Forward Pass
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len) #produces the final embeddings of sequences
        seq_output = F.normalize(seq_output, dim=1) #Normalizes the sequence output embeddings

        # Remove sequences with the same next item
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))
        #Compute Contrastive Losses
        loss_seq_item = self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)
        loss_seq_seq = self.seq_seq_contrastive_task(seq_output, same_pos_id, interaction)
        #Combine and Return Loss
        loss = loss_seq_item + self.lam * loss_seq_seq
        return loss

    #This method is designed to handle various phases of model training, such as pretraining and fine-tuning to find losses
    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)

        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        #Model Forward Pass
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        
        #Adjusting Test Item Embeddings
        test_item_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_item_emb = test_item_emb + self.item_embedding.weight
        #Normalizing Embeddings
        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(test_item_emb, dim=1)
        #Computing Similarity and Loss
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        return loss

    #the goal is to make predictions based on item sequences
    def full_sort_predict(self, interaction):
        #Extract Item Sequences and Lengths
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        #Get Item Embeddings and Compute Sequence Output
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        
        #Get Test Item Embeddings
        test_items_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft': #Adjust Test Item Embeddings if in 'transductive_ft' Stage
            test_items_emb = test_items_emb + self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=-1) #Norimalize the embeddings 
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        #Compute Scores
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
