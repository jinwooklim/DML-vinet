# python2.7
import os
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from tensorboardX import SummaryWriter

import os
from utils import tools
from utils import se3qua

import FlowNetC


from PIL import Image
import numpy as np

import flowlib

from PIL import Image

import csv
import time



class MyDataset:
    
    def __init__(self, base_dir, sequence):
        self.base_dir = base_dir
        self.sequence = sequence
        self.base_path_img = self.base_dir + self.sequence + '/cam0/data/'
        
        
        self.data_files = os.listdir(self.base_dir + self.sequence + '/cam0/data/')
        self.data_files.sort()
        
        ## relative camera pose
        self.trajectory_relative = self.read_R6TrajFile('/vicon0/sampled_relative_R6.csv')
        
		## abosolute camera pose (global)
        self.trajectory_abs = self.readTrajectoryFile('/vicon0/sampled.csv')
        
		## imu
        self.imu = self.readIMU_File('/imu0/data.csv')
        
        self.imu_seq_len = 5
   
    def readTrajectoryFile(self, path):
        traj = []
        with open(self.base_dir + self.sequence + path) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                parsed = [float(row[1]), float(row[2]), float(row[3]), 
                          float(row[4]), float(row[5]), float(row[6]), float(row[7])]
                traj.append(parsed)
                
        return np.array(traj)
    
    def read_R6TrajFile(self, path):
        traj = []
        with open(self.base_dir + self.sequence + path) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                parsed = [float(row[1]), float(row[2]), float(row[3]), 
                          float(row[4]), float(row[5]), float(row[6])]
                traj.append(parsed)
                
        return np.array(traj)
    
    def readIMU_File(self, path):
        imu = []
        count = 0
        with open(self.base_dir + self.sequence + path) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                if count == 0:
                    count += 1
                    continue
                parsed = [float(row[1]), float(row[2]), float(row[3]), 
                          float(row[4]), float(row[5]), float(row[6])]
                imu.append(parsed)
                
        return np.array(imu)

    '''
    def getTrajectoryAbs(self, idx):
        return self.trajectory_abs[idx]
    '''

    def getTrajectoryAbs(self, idx, batch, timesteps):
        #for f in range(batch):
        #    print(idx+(f*timesteps))
        return np.asarray([self.trajectory_abs[idx+(f * timesteps)] for f in range(batch)]) # all has same [idx] value
        

    def getTrajectoryAbsAll(self):
        return self.trajectory_abs
    
    def getIMU(self):
        return self.imu
    
    def __len__(self):
        return len(self.trajectory_relative)
    
    def load_img_bat(self, idx, batch, timesteps):
        batch_X = []
        batch_X2 = []
        batch_Y = []
        batch_Y2 = []
        
        for batch_idx in range(batch):
            timesteps_x = []
            timesteps_imu = []
            #print("batch : ", batch_idx)
            for timestep_idx in range(timesteps):
                #print(idx + timestep_idx, idx+1 + timestep_idx)
                x_data_np_1 = np.array(Image.open(self.base_path_img + self.data_files[idx + timestep_idx]))
                x_data_np_2 = np.array(Image.open(self.base_path_img + self.data_files[idx+1 + timestep_idx]))

                ## 3 channels
                x_data_np_1 = np.array([x_data_np_1, x_data_np_1, x_data_np_1])
                x_data_np_2 = np.array([x_data_np_2, x_data_np_2, x_data_np_2])

                X = np.array([x_data_np_1, x_data_np_2])
                timesteps_x.append(X)
                
                #print(idx-self.imu_seq_len+1+timestep_idx, idx+1+timestep_idx)
                tmp = np.array(self.imu[idx-self.imu_seq_len+1+timestep_idx : idx+1+timestep_idx])
                timesteps_imu.append(tmp)
            
            idx = idx + timesteps

            timesteps_x = np.array(timesteps_x)
            timesteps_imu = np.array(timesteps_imu)
        
            #X = Variable(torch.from_numpy(timesteps_x).type(torch.FloatTensor).cuda())    
            #X2 = Variable(torch.from_numpy(timesteps_imu).type(torch.FloatTensor).cuda())    
            X = timesteps_x
            X2 = timesteps_imu
             
            ## F2F gt
            #Y = Variable(torch.from_numpy(self.trajectory_relative[idx+1:idx+1+timesteps]).type(torch.FloatTensor).cuda())
            Y = self.trajectory_relative[idx+1 : idx+1+timesteps]

            ## global pose gt
            #Y2 = Variable(torch.from_numpy(self.trajectory_abs[idx+1:idx+1+timesteps]).type(torch.FloatTensor).cuda())
            Y2 = self.trajectory_abs[idx+1 : idx+1+timesteps]

            batch_X.append(X)
            batch_X2.append(X2)
            batch_Y.append(Y)
            batch_Y2.append(Y2)
        
        batch_X = np.asarray(batch_X)
        batch_X2 = np.asarray(batch_X2)
        batch_Y = np.asarray(batch_Y)
        batch_Y2 = np.asarray(batch_Y2)

        batch_X = Variable(torch.from_numpy(batch_X).type(torch.FloatTensor).cuda())
        batch_X2 = Variable(torch.from_numpy(batch_X2).type(torch.FloatTensor).cuda())
        batch_Y = Variable(torch.from_numpy(batch_Y).type(torch.FloatTensor).cuda())
        batch_Y2 = Variable(torch.from_numpy(batch_Y2).type(torch.FloatTensor).cuda())
        
        #print(batch_X.shape)
        #print(batch_X2.shape)
        #print(batch_Y.shape)
        #print(batch_Y2.shape)
        #exit()
        return batch_X, batch_X2, batch_Y, batch_Y2 #return X, X2, Y, Y2

    
    
class Vinet(nn.Module):
    def __init__(self):
        super(Vinet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=24589,#49152,#24576, 
            hidden_size=1024,#64, 
            num_layers=2,
            batch_first=True)
        self.rnn.cuda()
        
        self.rnnIMU = nn.LSTM(
            input_size=6, 
            hidden_size=6,
            num_layers=2,
            batch_first=True)
        self.rnnIMU.cuda()
        
        self.linear1 = nn.Linear(1024, 128)
        self.linear2 = nn.Linear(128, 6)
        self.linear1.cuda()
        self.linear2.cuda()
        
        checkpoint = None
        checkpoint_pytorch = '/notebooks/model/FlowNet2-C_checkpoint.pth.tar'
        #checkpoint_pytorch = '/notebooks/data/model/FlowNet2-SD_checkpoint.pth.tar'
        if os.path.isfile(checkpoint_pytorch):
            checkpoint = torch.load(checkpoint_pytorch,\
                                map_location=lambda storage, loc: storage.cuda(0))
            best_err = checkpoint['best_EPE']
        else:
            print('No checkpoint')

        self.flownet_c = FlowNetC.FlowNetC(batchNorm=False)
        self.flownet_c.load_state_dict(checkpoint['state_dict'])
        self.flownet_c.cuda()

    def forward(self, image, imu, xyzQ):
        batch_size, timesteps, _, C, H, W = image.size() # [batch, timesteps, 2, channel, Height, Width]
         
        ## Input1: Feed image pairs to FlownetC
        ##c_in = image.view(batch_size, timesteps * C, H, W)
        c_out_list = []
        for idx_b in range(batch_size):
            each_batch_image = image[idx_b,...]
            c_in = each_batch_image.view(timesteps, 2 * C, H, W)
            c_out = self.flownet_c(c_in)
            c_out_list.append(c_out)
        c_out = torch.stack(c_out)
         
        ## Input2: Feed IMU records to LSTM
        imu_out_list = []
        for idx_b in range(batch_size):
            each_batch_imu = imu[idx_b,...]
            imu_out, (imu_n, imu_c) = self.rnnIMU(each_batch_imu)
            imu_out = imu_out[:, -1, :]
            imu_out = imu_out.unsqueeze(1)
            imu_out_list.append(imu_out)
        imu_out = torch.stack(imu_out_list)
        imu_out = imu_out.view(batch_size, timesteps, -1) # (batch, timesteps, 6)
        
        ## Combine the output of input1 and 2 and feed it to LSTM
        ##r_in = c_out.view(batch_size, timesteps, -1)
        r_in = c_out.view(batch_size, timesteps, -1) # (batch, timesteps, 24576)
        cat_out = torch.cat((r_in, imu_out), 2) # (batch, timesteps, input_size)
        xyzQ = xyzQ.repeat(1, timesteps, 1) # (batch, timesteps, 7)
        
        cat_out = torch.cat((cat_out, xyzQ), 2) # (batch_size, timesteps, input_size)
        
        ## Feed concatenate data to Main LSTM stream
        r_out, (h_n, h_c) = self.rnn(cat_out) # (batch_size, timesteps, 1024)
        l_out1 = self.linear1(r_out) # (batch_size, timesteps, 128)
        l_out2 = self.linear2(l_out1) # (batch_size, timesteps, 6)
        return l_out2
    
    
def model_out_to_flow_png(output):
    out_np = output[0].data.cpu().numpy()

    #https://gitorchub.com/DediGadot/PatchBatch/blob/master/flowlib.py
    out_np = np.squeeze(out_np)
    out_np = np.moveaxis(out_np,0, -1)

    im_arr = flowlib.flow_to_image(out_np)
    im = Image.fromarray(im_arr)
    im.save('test.png')


def train():
    epoch = 20
    batch = 2
    timesteps = 4
    model = Vinet()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    writer = SummaryWriter()
    
    model.train()

    mydataset = MyDataset('/notebooks/EuRoC_modify/', 'V1_01_easy')
    #criterion  = nn.MSELoss()
    criterion  = nn.L1Loss(size_average=False)
    
    start = 5
    end = len(mydataset)-batch
    batch_num = (end - start) #/ batch
    startT = time.time() 
    abs_traj = None
    
    with tools.TimerBlock("Start training") as block:
        for k in range(epoch):
            for i in range(start, end):#len(mydataset)-1):
                data, data_imu, target_f2f, target_global = mydataset.load_img_bat(i, batch, timesteps)
                data, data_imu, target_f2f, target_global = \
                    data.cuda(), data_imu.cuda(), target_f2f.cuda(), target_global.cuda()
                #print(data.shape)
                #print(data_imu.shape)
                #print(target_f2f.shape)
                #print(target_global.shape)
                #exit() 
                
                optimizer.zero_grad()
                
                '''            
                if i == start:
                    ## load first SE3 pose xyzQuaternion
                    abs_traj = mydataset.getTrajectoryAbs(start)
                    
                    abs_traj_input = np.expand_dims(abs_traj, axis=0)
                    abs_traj_input = np.expand_dims(abs_traj_input, axis=0)
                    abs_traj_input = Variable(torch.from_numpy(abs_traj_input).type(torch.FloatTensor).cuda()) 
                '''
                
                if i == start:
                    ## load first SE3 pose xyzQuaternion
                    abs_traj = mydataset.getTrajectoryAbs(start, batch, timesteps) # (batch, 7)
                    abs_traj_input = np.expand_dims(abs_traj, axis=1) # (batch, 1, 7)
                    abs_traj_input = Variable(torch.from_numpy(abs_traj_input).type(torch.FloatTensor).cuda()) 

                ## Forward
                output = model(data, data_imu, abs_traj_input)
                
                ## Accumulate pose
                numarr = output.data.cpu().numpy()
              
                ## SE part 
                abs_traj_list = []
                for idx in range(batch):
                    print(abs_traj.shape, numarr.shape)
                    print(abs_traj[idx,...].shape, numarr[idx,...].shape)
                    abs_traj_temp = se3qua.accu(abs_traj[idx,...], numarr[idx,...])
                    abs_traj_list.append(abs_traj_temp)
                abs_traj = np.asarray(abs_traj_list) # (4,7)
                exit() 
                abs_traj_input = np.expand_dims(abs_traj, axis=1)
                abs_traj_input = Variable(torch.from_numpy(abs_traj_input).type(torch.FloatTensor).cuda()) # (batch,1,7)

                ## (F2F loss) + (Global pose loss)
                ## Global pose: Full concatenated pose relative to the start of the sequence
                loss = criterion(output, target_f2f) + criterion(abs_traj_input, target_global)

                loss.backward()
                optimizer.step()

                avgTime = block.avg()
                remainingTime = int((batch_num*epoch -  (i + batch_num*k)) * avgTime)
                rTime_str = "{:02d}:{:02d}:{:02d}".format(int(remainingTime/60//60), 
                                                          int(remainingTime//60%60), 
                                                          int(remainingTime%60))

                block.log('Train Epoch: {}\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}, TimeAvg: {:.4f}, Remaining: {}'.format(
                    k, i , batch_num,
                    100. * (i + batch_num*k) / (batch_num*epoch), loss.data[0], avgTime, rTime_str))
                
                writer.add_scalar('loss', loss.data[0], k*batch_num + i)
                
            check_str = 'checkpoint_{}.pt'.format(k)
            torch.save(model.state_dict(), check_str)
            
    #torch.save(model, 'vinet_v1_01.pt')
    #model.save_state_dict('vinet_v1_01.pt')
    torch.save(model.state_dict(), 'vinet_v1_01.pt')
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


def test():
    #checkpoint_pytorch = '/notebooks/vinet/vinet_v1_01.pt'
    checkpoint_pytorch = '/notebooks/vinet/checkpoint_0.pt'
    if os.path.isfile(checkpoint_pytorch):
        checkpoint = torch.load(checkpoint_pytorch,\
                            map_location=lambda storage, loc: storage.cuda(0))
        #best_err = checkpoint['best_EPE']
    else:
        print('No checkpoint')
    
    model = Vinet()
    model.load_state_dict(checkpoint)  
    model.cuda()
    model.eval()
    #mydataset = MyDataset('/notebooks/EuRoC_modify/', 'V2_01_easy')
    datapath = "V2_01_easy"
    mydataset = MyDataset('/notebooks/data/', datapath)
    
    err = 0
    ans = []
    abs_traj = None
    start = 5
    for i in range(start, len(mydataset)-1):
    #for i in range(start, 100):
        data, data_imu, target, target2 = mydataset.load_img_bat(i, 1)
        data, data_imu, target, target2 = data.cuda(), data_imu.cuda(), target.cuda(), target2.cuda()

        if i == start:
            ## load first SE3 pose xyzQuaternion
            abs_traj = mydataset.getTrajectoryAbs(start)
            abs_traj = np.expand_dims(abs_traj, axis=0)
            abs_traj = np.expand_dims(abs_traj, axis=0)
            abs_traj = Variable(torch.from_numpy(abs_traj).type(torch.FloatTensor).cuda()) 
                    
        output = model(data, data_imu, abs_traj)
        
        err += float(((target - output) ** 2).mean())
        
        output = output.data.cpu().numpy()

        xyzq = se3qua.se3R6toxyzQ(output)
                
        abs_traj = abs_traj.data.cpu().numpy()[0]
        numarr = output
        
        abs_traj = se3qua.accu(abs_traj, numarr)
        abs_traj = np.expand_dims(abs_traj, axis=0)
        abs_traj = np.expand_dims(abs_traj, axis=0)
        abs_traj = Variable(torch.from_numpy(abs_traj).type(torch.FloatTensor).cuda()) 
        
        ans.append(xyzq)
        print(xyzq)
        print('{}/{}'.format(str(i+1), str(len(mydataset)-1)) )
        
    print('err = {}'.format(err/(len(mydataset)-1)))  
    trajectoryAbs = mydataset.getTrajectoryAbsAll()
    print(trajectoryAbs[0])
    x = trajectoryAbs[0].astype(str)
    x = ",".join(x)
    
    #with open('/notebooks/EuRoC_modify/V2_01_easy/vicon0/sampled_relative_ans.csv', 'w+') as f:
    with open('/notebooks/vinet/%s_sampled_relative_ans.csv'%datapath, 'w') as f:
        tmpStr = x
        f.write(tmpStr + '\n')        
        
        for i in range(len(ans)-1):
            tmpStr = ans[i].astype(str)
            tmpStr = ",".join(tmpStr)
            print(tmpStr)
            print(type(tmpStr))
            f.write(tmpStr + '\n')      
   
    
def main():
    train()
          
    #test()


if __name__ == '__main__':
    main()
