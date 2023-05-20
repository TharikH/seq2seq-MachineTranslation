import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Scoring function, which is used to calculate how many words in the batch is getting 100% word match
# This function is used in calculating accuracy
def scoring(y_dash , y):
    num_sample,seq_len = y.shape
    score = torch.sum(torch.sum(y_dash == y,axis = 1) == seq_len)
    return score

def testAccuracy(model, test_loader):
    model.eval()
    test_score = 0
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    for test_input, test_target in test_loader:
        test_input = test_input.to(device)
        test_target = test_target.to(device)
        test_output,_ = model.forward(test_input, None,False,False)

        acc_output = F.softmax(test_output,dim=2)
        acc_output = torch.argmax(acc_output,dim=2)
        acc_output = acc_output.T
        test_score += scoring(acc_output,test_target)


        test_output = test_output.permute(1, 0, 2)
        expected = F.one_hot(test_target,num_classes = 72).float()

        test_output = test_output.reshape(-1, 72)

        expected = expected.reshape(-1,72)


        loss = criterion(test_output, expected)
        test_loss += loss.item()
        
    print(f'test loss => {test_loss/len(test_loader)} \ntest_acc => {test_score/len(test_loader.dataset)}')
    return test_loss/len(test_loader),test_score/len(test_loader.dataset)

def runModel(model, data_loader, val_loader ,epochs, beam):
        
    # Set all training parameters
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # set model to train mode
    model.train()
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []

    # Do training in epoch fashion
    for epoch in tqdm(range(epochs)):
        total_loss=0
        train_loss = 0
        train_score = 0
        val_score = 0
        val_loss = 0

        # use data loader and enumerate each of data for training in batchwise
        for i, (source, target) in enumerate(data_loader):


            source = source.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output,_ = model.forward(source, target, epoch < epochs/2, False)

            # In order to do loss calc, first need to convert target to one-hot and make predicted in probabilistic manner
            output = output.permute(1, 0, 2)
            expected = F.one_hot(target,num_classes = 72).float()

            # make predicted and target in same dimension
            output = output.reshape(-1, 72)
            expected = expected.reshape(-1,72)

            # Calculate loss
            loss = criterion(output, expected)

            # Calculate gradients
            loss.backward()  

            # Clip gradiens, so will not explode
            nn.utils.clip_grad_norm_(model.parameters(),1)
            
            #update parameters
            optimizer.step()  

        # Calculate validation accuracy and losses => Same process as training, but here no updation of gradients
        with torch.no_grad():
            model.eval()
            for val_input, val_target in val_loader:
                val_input = val_input.to(device)
                val_target = val_target.to(device)
                val_output,_ = model.forward(val_input, None,False,beam)

                acc_output = F.softmax(val_output,dim=2)
                acc_output = torch.argmax(acc_output,dim=2)
                acc_output = acc_output.T
                val_score += scoring(acc_output,val_target)


                val_output = val_output.permute(1, 0, 2)
                expected = F.one_hot(val_target,num_classes = 72).float()

                val_output = val_output.reshape(-1, 72)

                expected = expected.reshape(-1,72)


                loss = criterion(val_output, expected)
                val_loss += loss.item()

        # Calculate training accuracy and losses
        with torch.no_grad():
            model.eval()
            for train_input, train_target in data_loader:
                train_input = train_input.to(device)
                train_target = train_target.to(device)
                train_output,_ = model.forward(train_input, None,False,beam)

                acc_output = F.softmax(train_output,dim=2)
                acc_output = torch.argmax(acc_output,dim=2)
                acc_output = acc_output.T
                train_score += scoring(acc_output,train_target)


                train_output = train_output.permute(1, 0, 2)
                expected = F.one_hot(train_target,num_classes = 72).float()

                train_output = train_output.reshape(-1, 72)

                expected = expected.reshape(-1,72)


                loss = criterion(train_output, expected)
                train_loss += loss.item()
            model.train()



        print(f'epoch {epoch}')
        print(f'train loss => {train_loss/len(data_loader)} \ntrain_acc => {train_score/len(data_loader.dataset)}')
        print(f'valid loss => {val_loss/len(val_loader)} \nvalid_acc => {val_score/len(val_loader.dataset)}')
        train_loss_list.append(train_loss/len(data_loader))
        val_loss_list.append(val_loss/len(val_loader))
        train_accuracy_list.append(train_score/len(data_loader.dataset))
        val_accuracy_list.append(val_score/len(val_loader.dataset))

    return train_loss_list,val_loss_list,train_accuracy_list,val_accuracy_list 

def getTicks(test_input, acc_output, data_index, english_index_dict, malayalam_index_dict):
    
    # put ticks into array by iterating test input and predicted of test and converting it into english-malayalam combination
    x_t = []
    y_t = []
    for i in range(len(test_input[data_index])):
        if(test_input[data_index][i].item() != 0 and test_input[data_index][i].item() != 1 and test_input[data_index][i].item() != 2):
            x_t.append(english_index_dict[test_input[data_index][i].item()])

    for i in range(len(acc_output[data_index])):
        if(acc_output[data_index][i].item() != 0 and acc_output[data_index][i].item() != 1 and acc_output[data_index][i].item() != 2):
            y_t.append(malayalam_index_dict[acc_output[data_index][i].item()])
            
    return x_t, y_t

def plotHeatMap(test_input,acc_output,w,english_index_dict, malayalam_index_dict,num_plots = 12):
    
    # Create subplots
    fig, ax = plt.subplots(4, 3,figsize=(20, 20))
    _ = plt.setp(ax)
    for data_index in range(num_plots):
        
        # get ticks
        x_t, y_t = getTicks(test_input, acc_output, data_index,english_index_dict, malayalam_index_dict)
        
        # w contains al attention weights for each batches, so take weights batch by batch
        a = w[:,:,data_index]
        a = a.detach().cpu().numpy()
        
        #remove start and end token's weights 
        a = a[1:len(y_t)+1,2:len(x_t)+2] 
        
        # plot 3 subplots per row
        plt.sca(ax[data_index//3,data_index%3])
        
        # Heat map using sns library
#         sns.heatmap(a)
#         plt.xticks(np.arange(0.5, len(x_t)+0.5), x_t)

        # Anjali dataset to be used as correct font for malayalam in ticks for plot
#         mal_font = FontProperties(fname = '/kaggle/input/anjali/AnjaliOldLipi-Regular.ttf')
#         plt.yticks(np.arange(0.5, len(y_t)+0.5), y_t,fontproperties= mal_font)
        
        
        # Heat map using matplotlib
        plt.imshow(a, interpolation='nearest')
        plt.colorbar()
        plt.xticks(np.arange(0, len(x_t)), x_t)
        
        # Anjali dataset to be used as correct font for malayalam in ticks for plot
        mal_font = FontProperties(fname = './AnjaliOldLipi-Regular.ttf')
        
        plt.yticks(np.arange(0, len(y_t)), y_t,fontproperties= mal_font)
        
        plt.xlabel('English')
        plt.ylabel('Malayalam')
        plt.title(f'test image {data_index + 1}')
        
        
#         plt.show()
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    # Save image to store it into wandb logs
    image = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.show()
    
    return image

