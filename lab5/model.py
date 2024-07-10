
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        '''
        TODO:
        conv1: 输入维度，输出维度:32，kernal_size:3, stride:2, padding:1
        conv2:输入维度：32，输出维度:32，kernal_size:3, stride:2, padding:1
        conv3:输入维度：32，输出维度:32，kernal_size:3, stride:2, padding:1
        conv4:输入维度：32，输出维度:32，kernal_size:3, stride:2, padding:1
        lstm:(LSTMCell)，输入维度：32*6*6，输出维度512
        critic：输入512，输出1
        actor：输入512，输出action_space
        '''
        


        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):

        '''
        TODO: 完成前向过程
        要求：返回actor值，critic值，hx值，cx值
        '''
        return None


