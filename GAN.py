from GAN.dataset import CompositionDataset
from torch.utils.data import DataLoader
import torch
from utils.tool import set_random_seed, model_summary
from pymatgen.core.composition import Composition
from GAN.gan import DiscriminatorLinear, GeneratorLinear
from tqdm.auto import tqdm
from heas import *
from utils.tool import *
import pandas as pd
import os

def generate_samples(
        model_path, 
        epochs=1, 
        batch_size=2500, 
        device="cuda",
        elements_list=None,
        g_condi=None
        ):
        """ Helper function to iteratively generate valid samples """  
        comp_list = []
        model = torch.load(model_path, weights_only=False)
        model.to(device)
        model.eval()
        
        for i in range(epochs):
          
            fake_noise = torch.randn(batch_size, config['z_dim']).float().to(device)
            if g_condi is not None:
                # 条件编码
                y = torch.zeros((batch_size, 2)).to(device)
                y[:, 0] = 1
                fake_gen = model(fake_noise, y).cpu().detach()
            else:
                fake_gen = model(fake_noise).cpu().detach()
            comp_list += feature2composition(fake_gen, elements_list)
        
        data = pd.DataFrame({'composition': comp_list})
        data = data.dropna()
        print("删除None后：{}/{}".format(data.shape[0], len(comp_list)))
        # data = data.drop_duplicates()
        # print("删除重复后：{}/{}".format(data.shape[0], len(comp_list)))
        print("元素统计：")
        data['element_num'] = data['composition'].apply(lambda comp: len(comp.elements))
        print(data['element_num'].value_counts())
        data = data[data['element_num']>1]
        data = data[data['element_num']<11]
        print("2~10之间的HEAs：{}/{}".format(data.shape[0], len(comp_list)))
        data.to_csv("./output/sample.csv", index=False)


def train(config, elements_list, file_name, condi=False, c_value=None):
    data = pd.read_csv(f"./data/{file_name}")
    train_data = data.sample(frac=0.9, random_state=config["seed"])
    test_data = data.drop(train_data.index)
    train_data.to_csv("./data/gan_train.csv", index=False)
    test_data.to_csv("./data/gan_test.csv", index=False)
    train_dataset = CompositionDataset(train_data, elements_list, condi, c_value)
    test_dataset = CompositionDataset(test_data, elements_list, condi, c_value)
    print("train length:", len(train_dataset), "test length:", len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                              pin_memory=True, num_workers=1, persistent_workers=True, prefetch_factor=24
                            ) 
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, 
                             pin_memory=True, num_workers=1, persistent_workers=True, prefetch_factor=24
                            )

    discriminator = DiscriminatorLinear(condi=condi)

    generator = GeneratorLinear(config['z_dim'], condi=condi)
    device = config['device']
    print('Model architectures')
    print("generator")
    model_summary(generator)
    print("discriminator")
    model_summary(discriminator)

    d_optimizer = torch.optim.Adam(discriminator.parameters(), config['lr'])
    g_optimizer = torch.optim.Adam(generator.parameters(), config['lr'])
    print("\n\n Starting GAN training\n\n")
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    generator.train()
    discriminator.train()
    epochs = config['epochs']
   
    for epoch in range(epochs): 
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, data in loop:
            if isinstance(data, list):
                x_real = data[0]
                y = data[1]
                y = y.to(device)
            else:
                y = None
                x_real = data
            x_real = x_real.to(device)
            for j in range(config['n_critic']):
                # 计算真实数据的损失
                real_out = discriminator(x_real, y)  # 将真实数据放入判别器中
                loss_real = -torch.mean(real_out)

                # 计算假数据的损失
                z = torch.randn(x_real.size(0), config['z_dim']).to(device)

                x_fake = generator(z, y)
                fake_out = discriminator(x_fake.detach(), y)
                loss_fake = torch.mean(fake_out)
                # Clip weights of discriminator
                # for p in discriminator.parameters():
                #     p.data.clamp_(-0.01, 0.01)

                # 梯度惩罚
                alpha = torch.rand(x_real.size()[0], 1).expand(x_real.size()).to(device)
                one = torch.tensor([1]).to(device)
                x_hat = (alpha * x_real + (one - alpha) * x_fake)
                pred_hat = discriminator(x_hat, y)
                gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, 
                                grad_outputs=torch.ones(pred_hat.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradient_penalty = ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
                ###########

                # 损失函数和优化
                d_loss = loss_real + loss_fake + config['lambda_penalty'] * gradient_penalty # 损失包括判真损失和判假损失
                d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
                d_loss.backward()  # 将误差反向传播
                d_optimizer.step()  # 更新参数
            
            z = torch.randn(x_real.size(0), config['z_dim']).to(device)
            x_fake = generator(z, y)
            fake_out = discriminator(x_fake, y)
            g_loss = -torch.mean(fake_out)       # 得到的假的图片与真实的图片的label的loss
            # bp and optimize
            g_optimizer.zero_grad()  # 梯度归0
            g_loss.backward()  # 进行反向传播
            g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(**{'D_loss': d_loss.item(), 'G_loss': g_loss.item()})
    
        # if (epoch+1) % 5 == 0:
        #     generate_samples(model_path='./saved_model/GAN/GAN_Conv.pt', batch_size=10000, epochs=1)
    torch.save(generator, os.path.join("./model", config['model_name'])) 
    print("\nFinished GAN training")

def train_model(config, elements_list, file_name, condi, c_value=None):
    time_1 = time.time()
    train(config, elements_list, file_name, condi, c_value)
    time_2 = time.time()
    print("training time: {}".format(time_2-time_1))

if __name__ == "__main__":
    elements_list = ['Fe', 'Mg', 'Al', 'Ti', 'Y', 'W', 'La', 'Cr', 'Co', 'C', 'V', 'Cu', 'Mn', 'Ce', 'B', 'Sn', 'Mo', 'Nb', 'Ni', 'Si', 'Gd', 'Zr']
    import time
    config = {'seed': 49, 'z_dim': 25, "n_critic": 3, 'batch_size': 32, 'lambda_penalty': 10, 
              'lr': 0.001, 'epochs': 12, 'model_name': 'CGAN_wt.pt', "file_name": 'wt.csv', 'condition': True,
              "c_value": 1.51, 'g_condi': [1, 0]} # ΔH 31 ,wt% 1.51
    if config['condition'] == False:
        config['g_condi'] = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU is available:"+str(torch.cuda.is_available())+", Quantity: "+str(torch.cuda.device_count())+'\n')
    print(f"Running on {device}")
    set_random_seed(config['seed'])
    config['device'] = device
    # time_1 = time.time()
    # train(config)
    train_model(config, elements_list, config['file_name'], config['condition'], config['c_value'])
    generate_samples(model_path=f"./model/{config['model_name']}", batch_size=1000, epochs=1, elements_list=elements_list, g_condi=config['g_condi'])
