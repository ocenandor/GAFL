import torch
from sklearn.metrics import f1_score


class DiceMetric(object):
    def __init__(self, n_classes=2):
        self.n_classes = n_classes

    @staticmethod
    def _input_format(predictions, targets):
        if len(targets.shape) == 4:
            predictions = torch.argmax(predictions, dim=1)
        if len(targets.shape) == 4:
            targets = targets.squeeze(dim=1)

        return predictions, targets

    def __call__(self, predictions, targets):
        predictions, targets = self._input_format(predictions, targets)

        one_hot_predictions = torch.nn.functional.one_hot(predictions, num_classes=self.n_classes)
        one_hot_predictions = one_hot_predictions.permute(0, 3, 1, 2).to(predictions.dtype)

        one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=self.n_classes)
        one_hot_targets = one_hot_targets.permute(0, 3, 1, 2).to(predictions.dtype)

        weights = (torch.ones(one_hot_targets.shape[:2]) / self.n_classes).to(predictions.device)

        intersections = torch.sum(one_hot_predictions * one_hot_targets, dim=(2, 3))
        unions = torch.sum(one_hot_predictions + one_hot_targets, dim=(2, 3))

        dice_coefficients = torch.sum(weights * (2 * intersections) / (unions + 1e-16), dim=1)

        return torch.mean(dice_coefficients)


class F1Score(object):
    def __init__(self, average='micro'):
        self.average = average

    @staticmethod
    def _input_format(predictions, targets):
        return torch.argmax(predictions, dim=1).cpu().tolist(), targets.cpu().tolist()

    def __call__(self, predictions, targets):
        predictions, targets = self._input_format(predictions, targets)
        return f1_score(y_true=targets, y_pred=predictions, average=self.average, pos_label=2)


class F1Score_nb(object):
    def __init__(self, average='macro'):
        self.average = average

    @staticmethod
    def _input_format(predictions, targets):
        idxs = ((targets == 0) + (targets == 2)).nonzero() # norm vs benign - 2 vs 0
        preds_ = predictions[idxs]
        preds_ = torch.argmax(preds_, dim=1).cpu().tolist()
        targets_ = targets[idxs].cpu().tolist()  
    
        return preds_, targets_

    def __call__(self, predictions, targets):
        predictions, targets = self._input_format(predictions, targets)
        return f1_score(y_true=targets, y_pred=predictions, average=self.average, pos_label=2)


"""
{'benign': 0, 'malignant': 1, 'normal': 2}
"""
# norm vs benign - 2 vs 0
class F1Score_NvsB(object):
    def __init__(self, average='binary'):
        self.average = average  # leaving macro but using 2 classes only with pos_label
        self.pos_label = 2 # norm vs benign - 2 vs 0
        self.class_label_1 = 2
        self.class_label_2 = 0

    @staticmethod
    def _input_format(predictions, targets, class_label_1, class_label_2):
        idxs = ((targets == class_label_1) + (targets == class_label_2)).nonzero() # norm vs benign - 2 vs 0
        
        #print(predictions)
        preds_ = predictions
        #print(preds_)

        preds_ = torch.argmax(preds_, dim=1).cpu()
        #print(preds_)

        #idxs = ((preds_ == class_label_1) + (preds_ == class_label_2)).nonzero() # norm vs benign - 2 vs 0
        #preds_ = preds_[idxs]
        
        #print(preds_)
        preds_ = torch.where(preds_ == torch.tensor(2),
                           1,
                           0)
        #print(preds_)
        preds_ = preds_[idxs].tolist()
        ##############################################
        targets_ = targets.cpu()
        #print(targets_)
        targets_ = torch.where(targets_ == torch.tensor(2),
                           1,
                           0)
        #print(targets_)                           
        targets_ =  targets_[idxs].tolist()
        
        
        #у тебя там авг считается по всему но тут ты убираешь же 
        return preds_, targets_

    def __call__(self, predictions, targets):       
        predictions, targets = self._input_format(predictions, targets, self.class_label_1, self.class_label_2)
        
        '''
        print()
        print(predictions)
        print(targets)
        print()
        '''
        
        return f1_score(y_true=targets, y_pred=predictions, average=self.average, pos_label=1) # WHY 0 ?

# benign vs malignant - 0 vs 1
class F1Score_BvsM(object):
    def __init__(self, average='binary'):
        self.average = average  # leaving macro but using 2 classes only with pos_label
        self.pos_label = 0 # benign vs malignant - 0 vs 1
        self.class_label_1 = 0
        self.class_label_2 = 1

    @staticmethod
    def _input_format(predictions, targets, class_label_1, class_label_2, label_one ):
        idxs = ((targets == class_label_1) + (targets == class_label_2)).nonzero() 
        
        #print(predictions)
        preds_ = predictions
        #print(preds_)

        preds_ = torch.argmax(preds_, dim=1).cpu()
        #print(preds_)

        #idxs = ((preds_ == class_label_1) + (preds_ == class_label_2)).nonzero() 
        #preds_ = preds_[idxs]
        
        #print(preds_)
        preds_ = torch.where(preds_ == torch.tensor(label_one),
                           1,
                           0)
        #print(preds_)
        preds_ = preds_[idxs].tolist()
        ##############################################
        targets_ = targets.cpu()
        #print(targets_)
        targets_ = torch.where(targets_ == torch.tensor(label_one),
                           1,
                           0)
        #print(targets_)                           
        targets_ =  targets_[idxs].tolist()
        
        
        #у тебя там авг считается по всему но тут ты убираешь же 
        return preds_, targets_

    def __call__(self, predictions, targets):       
        predictions, targets = self._input_format(predictions, targets, self.class_label_1, self.class_label_2, label_one=self.pos_label)
        
        '''
        print()
        print(predictions)
        print(targets)
        print()
        '''
        
        return f1_score(y_true=targets, y_pred=predictions, average=self.average, pos_label=0) # WHY 0 ?



'''
class F1Score_BvsM(object):
    def __init__(self, average='binary'):
        self.average = average  # leaving macro but using 2 classes only with pos_label
        self.pos_label = 0 # benign vs malignant - 0 vs 1
        self.class_label_1 = 0
        self.class_label_2 = 1

    @staticmethod
    def _input_format(predictions, targets, class_label_1, class_label_2):
        idxs = ((targets == class_label_1) + (targets == class_label_2)).nonzero() # benign vs malignant - 0 vs 1
        
        return torch.argmax(predictions, dim=1)[idxs].cpu().tolist(), targets[idxs].cpu().tolist()

    def __call__(self, predictions, targets):
        predictions, targets = self._input_format(predictions, targets, self.class_label_1, self.class_label_2)
        return f1_score(y_true=targets, y_pred=predictions, average=self.average, pos_label=self.pos_label)
'''     
        
        
        
        
# normal vs malignant - 2 vs 1
class F1Score_NvsM(object):
    def __init__(self, average='macro'):
        self.average = average  # leaving macro but using 2 classes only with pos_label
        self.pos_label = 2 # benign vs malignant - 0 vs 1
        self.class_label_1 = 2
        self.class_label_2 = 1

    @staticmethod
    def _input_format(predictions, targets, class_label_1, class_label_2):
        idxs = ((targets == class_label_1) + (targets == class_label_2)).nonzero() # benign vs malignant - 0 vs 1
        
        return torch.argmax(predictions, dim=1)[idxs].cpu().tolist(), targets[idxs].cpu().tolist()

    def __call__(self, predictions, targets):
        predictions, targets = self._input_format(predictions, targets, self.class_label_1, self.class_label_2)
        return f1_score(y_true=targets, y_pred=predictions, average=self.average, pos_label=self.pos_label)