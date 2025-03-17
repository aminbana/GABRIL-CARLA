import torch
class GazeToMask():
    def __init__(self, width=320, height=180, sigmas=[10,10,10,10], coeficients = [1,1,1,1]):
        self.width = height
        self.height = width
        assert len(sigmas) == len(coeficients)
        self.sigmas = sigmas
        self.coeficients = coeficients
        self.masks = self.initialize_mask()

    def generate_single_gaussian_tensor(self, map_width:int, map_height:int, mean_x:float, mean_y:float, sigma:float):
        x = torch.arange(map_width, dtype=torch.float32).unsqueeze(1).expand(map_width, map_height)
        y = torch.arange(map_height, dtype=torch.float32).unsqueeze(0).expand(map_width, map_height)
        # Calculate the Gaussian distribution for each element
        gaussian_tensor = (1 / (2 * torch.pi * sigma ** 2)) * torch.exp(
            -((x - mean_x) ** 2 + (y - mean_y) ** 2) / (2 * sigma ** 2))

        return gaussian_tensor

    def initialize_mask(self):
        temp_map = []
        # N = self.width
        for i in range(len(self.sigmas)):
            temp = self.generate_single_gaussian_tensor(2 * self.width, 2 * self.height, self.width - 1, self.height - 1, self.sigmas[i])
            temp = temp/ temp.max()
            temp_map.append(self.coeficients[i]*temp)

        temp_map = torch.stack(temp_map, 0)


        return temp_map

    def find_suitable_map(self, heightx2=640, widthx2=360, index=0, mean_x=0.5, mean_y=0.5):
        # returns a map such that the center of the gaussian is located at (mean_x, mean_y) of the map
        start_x, start_y = int((1 - mean_x) * widthx2 / 2), int((1 - mean_y) * heightx2 / 2)
        desired_map = self.masks[index][start_y:start_y + heightx2 // 2, start_x:start_x + widthx2 // 2]
        return desired_map

    def find_bunch_of_maps(self, means=[[0.5, 0.5]], offset_start=0):
        current_maps = torch.zeros([self.width, self.height])
        bunch_size = len(means)
        assert bunch_size + offset_start <= len(self.sigmas), f'The bunch is too long! It\'s length is {bunch_size}'
        widthx2 = self.width * 2
        heightx2 = self.height * 2
        for i in range(bunch_size):
            mean_x, mean_y = means[i][0], means[i][1]
            # mean_x, mean_y = 0, 0
            temp = self.find_suitable_map(widthx2, heightx2, i+offset_start, mean_x, mean_y)
            current_maps = current_maps + temp

        return current_maps / torch.max(current_maps)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    mask_maker = GazeToMask(sigmas=[20, 20, 20, 20], coeficients=[1, 1, 1, 1])
    temp = mask_maker.masks[2]/ mask_maker.masks[2].max()
    ans = mask_maker.find_bunch_of_maps()
    # print(mask_maker.masks.shape)
    # plt.imshow(temp)
    # plt.show()