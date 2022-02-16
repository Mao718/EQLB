import torch
import time
import numpy as np
# batch version
# @torch.jit.script
from tqdm import trange


class SyncMap_:
    def __init__(self, input_size, dimensions, adaptation_rate, noise=False):

        self.name = "SyncMap_gpu"
        self.organized = False
        self.space_size = 100
        self.dimensions = dimensions
        self.input_size = input_size
        # syncmap= np.zeros((input_size,dimensions))
        self.syncmap = np.random.rand(input_size, dimensions)
        self.synapses_matrix = np.zeros([input_size, input_size])
        self.adaptation_rate = adaptation_rate
        # self.syncmap= np.random.rand(dimensions, input_size)

        self.ims = []
        self.fps = 0

# def div(a: 'torch tensor',b:'torch tensor'):
#    return a/b


@torch.jit.script
def batch_inputGeneral(x, syncmap, lr=torch.tensor(0.01), space_size=torch.tensor(10)):# w(batch, variable, dim)

    data = x
    representation = syncmap
    plus = torch.where(x > 0.1, 1., 0.)
    minus = torch.where(x > 0.1, 0., 1.)
    start_normalize = torch.zeros(data.shape[0])
    for i in range(data.shape[1]):
        vplus = plus[:, i, :]

        vminus = minus[:, i, :]
        continue_p = torch.sum(vplus)
        continue_n = torch.sum(vminus)
        if continue_p <= 1:
            continue
        if continue_n <= 1:
            continue
        # print(i,'th vplus',vplus)   #OK
        plus_mass = torch.sum(vplus, 1)  # .sum(axis=1)
        plus_mass[plus_mass == 1] = 0
        minus_mass = torch.sum(vminus, 1)  # .sum(axis=1)
        minus_mass[minus_mass == 0] = 0

        mask_plus = plus_mass.clone()
        mask_plus[mask_plus > 1] = 1
        mask_plus = torch.unsqueeze(mask_plus, 1)
        mask_plus = torch.unsqueeze(mask_plus, 2)
        mask_plus = torch.tile(
            mask_plus, [1, representation.shape[1], representation.shape[2]])

        mask_minus = minus_mass.clone()
        mask_minus[mask_minus > 1] = 1
        mask_minus = torch.unsqueeze(mask_minus, 1)
        mask_minus = torch.unsqueeze(mask_minus, 2)
        mask_minus = torch.tile(
            mask_minus, [1, representation.shape[1], representation.shape[2]])

        non_zero_plus_mass = plus_mass.clone()
        non_zero_plus_mass[non_zero_plus_mass == 0] = 1
        non_zero_minus_mass = minus_mass.clone()
        non_zero_minus_mass[non_zero_minus_mass == 0] = 1

        center_plus_temp_map = torch.tile(torch.unsqueeze(
            vplus, 2), [1, 1, representation.shape[2]])
        # print(i,'th',vplus)
        #print(i,'th center_plus_temp_map',center_plus_temp_map)
        center_minus_temp_map = torch.tile(torch.unsqueeze(
            vminus, 2), [1, 1, representation.shape[2]])
        center_plus_mass_temp = torch.unsqueeze(non_zero_plus_mass, 1)
        center_plus_mass_temp = torch.tile(
            center_plus_mass_temp, [1, representation.shape[2]])
        center_minus_mass_temp = torch.unsqueeze(non_zero_minus_mass, 1)
        center_minus_mass_temp = torch.tile(
            center_minus_mass_temp, [1, representation.shape[2]])

        center_plus = torch.div(torch.sum(
            mask_plus*center_plus_temp_map*representation, 1), center_plus_mass_temp)
        #print(i,'th center_plus',center_plus)
        #print(i,"th center_plus",center_plus)

        # test_center=torch.jit.fork(mutiple_center,vplus,representation,plus_mass)
        # test_center=torch.jit.wait(test_center)
        #print(i,"th test_center",test_center)

        center_minus = torch.div(torch.sum(
            mask_minus*center_minus_temp_map*representation, 1), center_minus_mass_temp)
        #print(i,'th center_minus',center_minus)
        cdist_plus_t = torch.tile(torch.unsqueeze(center_plus, 1), [
            1, representation.shape[1], 1])
        cdist_minus_t = torch.tile(torch.unsqueeze(center_minus, 1), [
            1, representation.shape[1], 1])

        cdist_plus = torch.sqrt(
            torch.sum(torch.square(representation-cdist_plus_t), 2))
        cdist_minus = torch.sqrt(
            torch.sum(torch.square(representation-cdist_minus_t), 2))
        #print(i,'th cdist_minus',cdist_minus)
        cdist_plus[cdist_plus == 0] = 1
        cdist_minus[cdist_minus == 0] = 1

        center_plus_temp = torch.unsqueeze(center_plus, 1)
        center_plus_temp = torch.tile(
            center_plus_temp, [1, representation.shape[1], 1])

        center_minus_temp = torch.unsqueeze(center_minus, 1)
        center_minus_temp = torch.tile(
            center_minus_temp, [1, representation.shape[1], 1])

        cd_update_plus_temp = torch.tile(torch.unsqueeze(cdist_plus, 2), [
                                         1, 1, representation.shape[2]])
        cd_update_minus_temp = torch.tile(torch.unsqueeze(
            cdist_minus, 2), [1, 1, representation.shape[2]])

        plus_temp = torch.tile(torch.unsqueeze(vplus, 2), [
                               1, 1, representation.shape[2]])
        minus_temp = torch.tile(torch.unsqueeze(vminus, 2), [
                                1, 1, representation.shape[2]])
        # ---------------------
        mask_sum_plus = torch.sum(mask_plus, (1, 2))
        mask_sum_minus = torch.sum(mask_minus, (1, 2))
        mask_plus[mask_sum_plus == 0] = 0
        mask_plus[mask_sum_minus == 0] = 0
        mask_minus[mask_sum_plus == 0] = 0
        mask_minus[mask_sum_minus == 0] = 0
        # ------------
        #print(i,'th center_plus_temp_map',center_plus_temp_map)
        update_plus = mask_plus*center_plus_temp_map * \
            ((center_plus_temp-representation)/cd_update_plus_temp)
        # mask -> batch, variable, dim
        update_minus = mask_minus*center_minus_temp_map * \
            ((center_minus_temp-representation)/cd_update_minus_temp)
        #print(i,'th center_minus_temp_map',center_minus_temp_map)
        #print(i,'th update_minus',update_minus)

        #print(i,' th mask_plus',torch.sum(mask_plus,0))
        #print(i,' th mask_minus',torch.sum(mask_minus,0))
        update = update_plus-update_minus
        #print(i,' th update_plus',update_plus)
        #print(i,'th updata',update)
        temp = torch.sum(update, 2)
        temp = torch.sum(temp, 1)

        start_normalize[temp != 0] = 1.
        representation += lr*update

        maximun, _ = torch.max(representation, 2)
        maximun, _ = torch.max(maximun, 1)
        normalize = space_size/maximun
        normalize[start_normalize == 0] = 1
        #print(i,'th normalize',normalize)
        normalize = torch.unsqueeze(normalize, 1)
        normalize = torch.unsqueeze(normalize, 2)
        normalize = torch.tile(
            normalize, [1, representation.shape[1], representation.shape[2]])
        representation = representation*normalize
        #print(i,'th representation',representation)
        # history.append(representation.numpy()[0].copy())
        # print('representation',representation)
    return representation  # ,history


# @torch.jit.script
# def _input(x, syncmap):
#     #x = torch.tensor(syncmap)
#     rep = syncmap
#     rep = torch.unsqueeze(rep, 0)
#     rep = torch.tile(rep, (x.shape[0], 1, 1))
#     representation = torch.jit.fork(inputGeneral, x=x, syncmap=rep)
#     return torch.jit.wait(representation)
if __name__ == "__main__":
    print(torch.__version__)
    start = time.time()
    rep = torch.rand((4000, 10, 5))  # .cuda()
    y = torch.rand((4000, 10000, 10))  # .cuda()

    represent = batch_inputGeneral(y, rep)
    end = time.time()
    print(end-start)
