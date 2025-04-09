import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalContrastiveLoss_mm(nn.Module):
    def __init__(self, n_classes, device, d=0, learnable_map=None, summaryWriter=None):
        super().__init__()
        self.d = d
        self.n_distances = n_classes - 1 # n-1 distances between n classes
        self.device = device

        # creating the learnable tensor to learn distances
        self.__createLearnableTensor(learnable_map)

        # for logging
        self.writer = summaryWriter 
        if self.writer != None: # creating a custom graph layout in tensorboard to plot distances
            self.writer.add_custom_scalars(self.__getGraphLayout())

    def __createLearnableTensor(self, learnable_map):
        if learnable_map == None: # if a learnable param map is not provided, this creates one to match the format
            learnable_map = []
            for _ in range(self.n_distances):
                learnable_map.append(['learnable', None])

        self.distances_ori = torch.zeros(self.n_distances, device=self.device, dtype=torch.float64) # distances_ori keeps the original learnable param values
        learnable_indices = [] # to store the indexes that has learnable params in distance_ori

        # creating the fixed and learnable parameters/distances according to the map
        for i, (isFixed, value) in enumerate(learnable_map):
            if isFixed == 'learnable':
                self.distances_ori[i] = self.__inverse_softplus(0.5 + torch.rand(1) * 0.5) if value is None else self.__inverse_softplus(torch.tensor([value])) # if None: initialise between 0.5-1
                learnable_indices.append(i)
            elif isFixed == 'fixed':
                self.distances_ori[i] = self.__inverse_softplus(torch.tensor([value]))

        if learnable_indices.__len__() > 0:
            learnable_indices = torch.tensor(learnable_indices, device=self.device)
            self.learnables = nn.Parameter(self.distances_ori[learnable_indices])
        
            # creating a mask to indicate learnable distances among all distances
            self.mask_learnables = torch.zeros_like(self.distances_ori, dtype=torch.bool)
            self.mask_learnables[learnable_indices] = True

    def __contrastiveLoss(self, prediction, target, step=None):
        """
            step: for tensorboard logging purposes only
        """
        # replacing the original params/distance tensor with learnable distances according to the mask
        self.distances = self.distances_ori.clone()
        if hasattr(self, 'mask_learnables'):
            self.distances[self.mask_learnables] = self.learnables

        n_samples = prediction.size()[0]

        # calculates cosine similarity matrix
        cos_sim = F.cosine_similarity(prediction.unsqueeze(0), prediction.unsqueeze(1), dim=2)

        # taking label differences
        label_diff = torch.abs(target.unsqueeze(0) - target.unsqueeze(1)).float()

        # taking positive and negative samples conditioning on label distance d
        positives = label_diff <= self.d
        negatives = ~positives
        positives[torch.eye(n_samples).bool()] = False # to avoid taking data point itself as a positive pair

        pos_cossim, neg_cossim = cos_sim.clone(), cos_sim.clone()
        pos_cossim[~positives] = torch.inf # setting false of positive tensor to inf
        neg_cossim[~negatives] = -torch.inf # setting false of negative tensor to -inf

        # deriving the distance matrix
        pos_distances = F.softplus(self.distances)
        class_positions = torch.cumsum(torch.cat([torch.tensor([0.0]).to(self.device), pos_distances]), dim=0)
        distance_matrix = torch.abs(class_positions.unsqueeze(0) - class_positions.unsqueeze(1))
        
        # logging the learning distances
        self.__logDistances(step, pos_distances)

        # assigning the margins
        label_indices = target.unsqueeze(0).repeat(n_samples, 1).to(self.device)
        margins = distance_matrix[label_indices, label_indices.t()]
        margins[~negatives] = 0

        mean_n_pair_loss = torch.tensor([]).to(self.device) # to collect n-pair loss for each column in positive tensor
        loss_masks_2 = torch.tensor([]).to(self.device)

        for pos_col in pos_cossim.T: # comparing each column of positive tensor with all columns of negative tensor
            n_pair_loss = (-pos_col + neg_cossim.T + margins.T).T

            # creating masks on elements n-pair loss was calculated for accurate mean calculation, otherwise final loss will be too low 
            loss_mask1 = ~torch.isinf(n_pair_loss) # -inf indicates it's not a pos/neg pair, avoiding those;
            loss_mask2 = loss_mask1.sum(dim=1)
            n_pair_loss = F.relu(n_pair_loss)

            # Compute the row-wise mean of the masked elements
            n_pair_loss = (n_pair_loss * loss_mask1).sum(dim=1) / loss_mask2.clamp(min=1)  # Use clamp to avoid division by zero

            loss_masks_2 = torch.cat((loss_masks_2, loss_mask2.reshape(1,-1)), dim=0)
            mean_n_pair_loss = torch.cat((mean_n_pair_loss, n_pair_loss.reshape(1,-1)), dim=0)

        mean_n_pair_loss = (mean_n_pair_loss * loss_masks_2.bool()).sum(dim=0) / loss_masks_2.sum(dim=0).clamp(min=1)
        return mean_n_pair_loss.mean()
    
    def __getGraphLayout(self):
        layout = {
            "Class Distances": {
                "margin": ["Multiline", [f'C{i}-C{i+1}' for i in range(self.n_distances)]],
            }
        }
        return layout
    
    def __logDistances(self, step, distances):
        if self.writer != None and step != None:
            for i, dist_val in enumerate(distances):
                self.writer.add_scalar(f"C{i}-C{i+1}", dist_val, step)
    
    def __inverse_softplus(self, t):
        # to get the inverse of softplus when setting margins
        return torch.where(t > 20, t, torch.log(torch.exp(t) - 1))
    
    def forward(self, prediction, target, step=None):
        """
            step: for tensorboard logging purpose only
        """
        return self.__contrastiveLoss(prediction, target, step)
    

class OrdinalContrastiveLoss_sm(nn.Module):
    def __init__(self, n_classes, device, d=0, learnable_map=None, summaryWriter=None):
        super().__init__()
        self.d = d
        self.n_distances = n_classes - 1 # n-1 distances between n classes
        self.device = device
        self.writer = summaryWriter
        self.__createLearnableTensor(learnable_map)
    
    def __createLearnableTensor(self, learnable_map):
        if learnable_map == None:
            # creating the learnable tensor to learn distances
            self.distance = nn.Parameter(self.__inverse_softplus(0.5 + torch.rand(1, device=self.device) * 0.5)) # initialise between 0.5-1
        else:
            isFixed, value = learnable_map[0]
            if isFixed == 'learnable':
                self.distance = nn.Parameter(self.__inverse_softplus(0.5 + torch.rand(1, device=self.device) * 0.5) if value is None else self.__inverse_softplus(torch.tensor([value], device=self.device)))
            elif isFixed == 'fixed':
                self.distance = self.__inverse_softplus(torch.tensor([value], device=self.device))

    def __contrastiveLoss(self, prediction, target, step=None):
        """
            step: for tensorboard logging purposes only
        """
        # replacing the original params/distance tensor with learnable distances according to the mask

        n_samples = prediction.size()[0]

        # calculates cosine similarity matrix
        cos_sim = F.cosine_similarity(prediction.unsqueeze(0), prediction.unsqueeze(1), dim=2)

        # taking label differences
        label_diff = torch.abs(target.unsqueeze(0) - target.unsqueeze(1)).float()

        # taking positive and negative samples conditioning on label distance d
        positives = label_diff <= self.d
        negatives = ~positives
        positives[torch.eye(n_samples).bool()] = False # to avoid taking data point itself as a positive pair

        pos_cossim, neg_cossim = cos_sim.clone(), cos_sim.clone()
        pos_cossim[~positives] = torch.inf # setting false of positive tensor to inf
        neg_cossim[~negatives] = -torch.inf # setting false of negative tensor to -inf

        # deriving the distance matrix
        pos_distances = F.softplus(self.distance)
        self.writer.add_scalar("Margin", pos_distances, step)

        pos_distances = pos_distances.repeat(self.n_distances)
        class_positions = torch.cumsum(torch.cat([torch.tensor([0.0]).to(self.device), pos_distances]), dim=0)
        distance_matrix = torch.abs(class_positions.unsqueeze(0) - class_positions.unsqueeze(1))

        # assigning the margins
        label_indices = target.unsqueeze(0).repeat(n_samples, 1).to(self.device)
        margins = distance_matrix[label_indices, label_indices.t()]
        margins[~negatives] = 0

        mean_n_pair_loss = torch.tensor([]).to(self.device) # to collect n-pair loss for each column in positive tensor
        loss_masks_2 = torch.tensor([]).to(self.device)

        for pos_col in pos_cossim.T: # comparing each column of positive tensor with all columns of negative tensor
            n_pair_loss = (-pos_col + neg_cossim.T + margins.T).T

            # creating masks on elements n-pair loss was calculated for accurate mean calculation, otherwise final loss will be too low 
            loss_mask1 = ~torch.isinf(n_pair_loss) # -inf indicates it's not a pos/neg pair, avoiding those;
            loss_mask2 = loss_mask1.sum(dim=1)
            n_pair_loss = F.relu(n_pair_loss)

            # Compute the row-wise mean of the masked elements
            n_pair_loss = (n_pair_loss * loss_mask1).sum(dim=1) / loss_mask2.clamp(min=1)  # Use clamp to avoid division by zero

            loss_masks_2 = torch.cat((loss_masks_2, loss_mask2.reshape(1,-1)), dim=0)
            mean_n_pair_loss = torch.cat((mean_n_pair_loss, n_pair_loss.reshape(1,-1)), dim=0)

        mean_n_pair_loss = (mean_n_pair_loss * loss_masks_2.bool()).sum(dim=0) / loss_masks_2.sum(dim=0).clamp(min=1)
        return mean_n_pair_loss.mean()
    
    def __inverse_softplus(self, t):
        # to get the inverse of softplus when setting margins
        return torch.where(t > 20, t, torch.log(torch.exp(t) - 1))
        
    def forward(self, prediction, target, step=None):
        """
            step: for tensorboard logging purpose only
        """
        return self.__contrastiveLoss(prediction, target, step)