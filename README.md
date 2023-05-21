# Deep-Learning  Assignment 3
This project is an implementation of a Machine Transliteration using pytorch. It also uses wandb for logging the data like accuracy and losses. You can create customized Sequence-to-Sequence network and train with aksharantar dataset with different configurations of parameters, then visualize the losses, accuracies and other plots(like attention plots) via wandb.

## Two ways to use or train
 - By command line arguments using train.py
 - By using the dl-rnn.ipynb (recommended)

### 1) Command line method for training using train.py
Here we need to run train.py file and specify wandb project names and key. But this method will take lots of time if computer doesn't have GPU support. So recommended option is using .ipynb file. The details for training are given below : 
### Dependencies
 - python
 - numpy library
 - wandb library
 - torch library
 - tqdm library (for viewing the iterations)
 - heapq (for implementing beam search)
 - matplotlib (If you want to plot attention matrix)

Either download the above dependencies or run :  `pip install -r requirements.txt`

### Usage
First make sure the above dependencies have been installed in your system. Then to use this project, simply clone the repository and run the train.py file. To download the repository, type the below command in the command line interface of git:

`git clone https://github.com/TharikH/seq2seq-MachineTranslation`

You can also download the entire repository as a zip file from the Download ZIP option on the above button provided by github.


We can give different values for the parameters and this can be done by specifying it in the command line arguments. Different possible arguments and its corresponding values are shown in the table below:

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | cs22m058  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-e`, `--epochs` | 20 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 256 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  [ "cross_entropy"] |
| `-nel`, `--num_encoder_layers` | 3 | Number of hidden layers used in encoder. | 
| `-ndl`, `--num_decoder_layers` | 3 | Number of hidden layers used in decoder. |
| `-eh`, `--encoder_hidden_size` | 256 | Number of hidden neurons in Encoder. |
| `-dh`, `--decoder_hidden_size` | 256 | Number of hidden neurons in Decoder. |
| `-bd`, `--bidirectional` | True | whether bidirection needed or not : True or False |
| `-ee`, `--encoder_embedding` | 256 |Dimension for input embedding. |
| `-de`, `--decoder_embedding` | 256 |Dimension for output embedding. |
| `-d`, `--dropout` | 0.2 | percentage of dropout needed. |
| `-c`, `--cell_type` | lstm | choices = ["lstm","rnn","gru"] |
| `-bw`, `--beam_width` | 1 | beam search beam width. | 
| `-a`, `--use_attention` | False | Need Attention or not : True or False |
</br>

Here it is an example of how to do run with epoch = 15 and num_encoder_layers as 2 : `python train.py -e 15 -nel 2`.

</br>
After running the file as shown above, training and testing accuracies will be printed on the terminal. Also the plots for training, validation, testing losses and accuracies for each epochs will be logged onto the wandb project (that you provided or into default project), so you can see the plots and see whether that particular configuration worked or not (If attention is true, attention weights plot will be logged).
For first half, teacher forcing is on, on the second half of the epoch, normal learning happens.


Also on using wandb, it is needed to put the corresponding key of the user so it can be logged on to the users wandb project.
It can be changed by searching for `key = <your key>`. at starting of train.py, the key variable will be present.


### 2) Using the dl-rnn.ipynb
Here open the dl-rnn.ipynb file on colab or kaggle (kaggle preferred as initi is done for kaggle). Then upload the `aksharntar` dataset (also provide correct path to the preprocesing part) and also upload `AnjaliOldLipi-Regular.ttf` so as to visualise heatmaps of attention weights in malayalam (Both have been provided in the repository => both path should be clearly mentioned whenever asked). Then at starting itself, paste the wandb key, by looking for `KEY = ` statement after imports. Every code will have a Label telling what that code is supposed to do. like `ALL NECESSARY IMPORTS` label specifying all the neccessary imports are being done in the following code. The different sections (Labels and their corresponding codes) are as follows : 


`PREPROCESSING STEP` It contains all the preprocessing steps, like creating english-index dictionaries, malayalam-index dictionaries, then loading data, converting each character into index format to correct sequence length, specifying batch sizes for data. So if you wish to change batch size, add different dataset you have to do it here.

`ENCODER MODULE` Here I have defined the encoder class, all the logic of encoder, how processing happens in encoder etc. So if one wishes to change encoder logic, do it here.

`DECODER MODULE` Here I have defined the decoder class, all the logic of decoder, how processing happens in decoder etc. So if one wishes to change decoder logic, do it here.

`BEAM SEARCH MODULE` Here I have defined the Beam search class, it is kind of bfs search, but search for highest probability path for decoding.

`SEQUENCE TO SEQUENCE MODULE` Here the overall transliteration happens. It will call encoders and decoders and transliterate. If you want to change the logic, change here.

`TRAINING FOR 1 EXAMPLE CONFIGURATION` Here if you wish to run 1 example and see how training goes without logging into wandb, then run this code. In order to change the configuration just change values in the model defining statement as given below:
```python
model = Seq2Seq(
    encoder_hidden_dimension = 256, 
    decoder_hidden_dimension =256,
    encoder_embed_dimension = 256, 
    decoder_embed_dimension = 256, 
    bidirectional = True,
    encoder_num_layers = 3,
    decoder_num_layers = 2,
    cell_type = 'lstm', 
    dropout = 0.2,
    beam_width = 1,
    device = device,
    attention = False
)
model.to(device)
```
By changing the values, we can create new configuration and run on it.


`SAVE MODEL` Here if we run on sample example, and need to save our model, then just uncomment it and run it, it will be saved (path should be properly given).

`LOAD MODEL` Here if we saved the model and need to load model for further processing, then provide the path for the model by uncommenting this statements (by default these load and saves are commented).


`TESTING AND ATTENTION PLOTS` If we need to plot attention weights, then run these, it will log the plot also and shows the plot for model that have been loaded or ran at training before.

`WANDB RUN FUNCTION` For running wandb, this function will be called, it will return list of accruacies and losses for training and validation, which will be used by wandb to log. This run fucntion is same as training function specified above, but mainly suited for logging in wandb.

`RUN SINGLE EXAMPLE AND LOG TO WANDB` For running for 1 example and log all the data into wandb, run these code. project name should be appropriately given. here also need to specify everything.
```python
wandb.login(key = KEY)
wandb.init(project = 'dl-assignment-3-final')
model = Seq2Seq(
    encoder_hidden_dimension = 256, 
    decoder_hidden_dimension =256,
    encoder_embed_dimension = 256, 
    decoder_embed_dimension = 256, 
    bidirectional = True,
    encoder_num_layers = 3,
    decoder_num_layers = 3,
    cell_type = 'lstm', 
    dropout = 0.2,
    beam_width = 3,
    device = device,
    attention = False
)
```
change these to suite your configurations.

`WANDB SWEEPS TRAIN FUNCTION` Here wandb sweeps train function is defined, which is being used for sweeping.

`WANDB SWEEP CONFIGURATIONS AND RUN SWEEPS` For specifying the sweep configurations, change the default values of the sweep as you like. the default sweep value is given as:
```python
sweep_configuration = {
    'method': 'bayes',
    'name': 'dl-assignment-3-final',
    'metric': {
        'goal': 'maximize', 
        'name': 'validation_accuracy'
        },
    'parameters': {
        'input_embedding': {'values': [16,32,64,128,256]},
        'number_of_enc_layer': {'values': [1,2,3]},
        'number_of_dec_layer': {'values': [1,2,3]},
        'hidden_size': {'values': [16,32,64,256]},
        'cell_type': {'values': ['rnn','gru','lstm']},
        'dropout': {'values': [0.2,0.3]},
        'bidirectional' : {'values' : [True,False]},
        'beam_width' : {'values' : [1,3,5]}
     }
}
```

`TEST ACCURACY CALCULATIONS` Here just pass your test data and it will find the test accuracy of your model - need to find model first, then only run this. It will also log into your wandb project.

`STORE SAMPLE OUTPUT FOR TEST DATA` If you want to store the prediction into a csv file, run these code. Path should be appropriately given.


### Additional Resources and help
All the sweep details for choosing the hyperparameters, runs, sample images, and related plots can be viewed by visiting (it contains the entire report) : `https://api.wandb.ai/links/cs22m058/157ivzox`.

