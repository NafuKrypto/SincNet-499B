# This code computes d-vectors using a pre-trained model


import os
import soundfile as sf
import torch,gc
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from dnn_models import MLP
from dnn_models import SincNet as CNN
from data_io import ReadList,read_conf_inp,str_to_bool
import sys

#Record your file 
# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import soundfile as sf

import json
from sklearn.metrics.pairwise import cosine_similarity

# Sampling frequency
freq = 44100

# Recording duration
duration = 3
dict1={}
dict2={}
speakers=[]
for i in range (0,2):
    #print("h")
    input("Press enter to record  voice")
    print("speak now....")
    
    # Start recorder with the given values
    # of duration and sample frequency
    recording = sd.rec(int(duration * freq),
    				samplerate=freq, channels=1)
    
    # Record audio for the given number of seconds
    sd.wait()
    #taking input of speakers name
    print("Enter a speaker's labeling or name in string.Don't give the same label even if the speaker's  are the same")
    input_label = input()
    speakers.append(input_label)
    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    write("recording0.wav", freq, recording)
    
    # Convert the NumPy array to audio file
    wv.write("recording1.wav", recording, freq, sampwidth=2)
    
    data, samplerate = sf.read('recording1.wav')
    sf.write('test_deploy/'+speakers[i]+'.wav', data, samplerate, subtype='PCM_16')
    
    print("Done")
    ob = sf.SoundFile('test_deploy/'+speakers[i]+'.wav')
    print('Sample rate: {}'.format(ob.samplerate))
    print('Channels: {}'.format(ob.channels))
    print('Subtype: {}'.format(ob.subtype))


# Model to use for computing the d-vectors
for index in range (0,2):
    #generte scp file
   
    if index==0:
        
        file_scp=open('test_deploy/'+'new1.scp',"w")
        file_scp.write(speakers[index]+'.wav')
        print(speakers[index]) 
        scp_path='new1.scp'
        file_scp.close()
    if index==1:
        file_scp=open('test_deploy/'+'new2.scp',"w")
        file_scp.write(speakers[index]+'.wav')
        print(speakers[index]) 
        scp_path='new2.scp'
        file_scp.close()
    
    model_file= 'test_deploy/timit_360/model_raw.pkl' # This is the model to use for computing the d-vectors (it should be pre-trained using the speaker-id DNN)
    cfg_file=r'cfg/SincNet_TIMIT.cfg' # Config file of the speaker-id experiment used to generate the model
    te_lst=r'test_deploy/'+scp_path # List of the wav files to process
    out_dict_file=r'test_deploy/d_vect_new.npy' # output dictionary containing the a sentence id as key as the d-vector as value
    data_folder=r'test_deploy/'
    
    avoid_small_en_fr=True
    energy_th = 0.1  # Avoid frames with an energy that is 1/10 over the average energy
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = None
    
    # Reading cfg file
    options=read_conf_inp(cfg_file)
    
    
    #[data]
    pt_file=options.pt_file
    output_folder=options.output_folder
    
    #[windowing]
    fs=int(options.fs)
    cw_len=int(options.cw_len)
    cw_shift=int(options.cw_shift)
    
    #[cnn]
    cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
    cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
    cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
    cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
    cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
    cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
    cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
    cnn_act=list(map(str, options.cnn_act.split(',')))
    cnn_drop=list(map(float, options.cnn_drop.split(',')))
    
    
    #[dnn]
    fc_lay=list(map(int, options.fc_lay.split(',')))
    fc_drop=list(map(float, options.fc_drop.split(',')))
    fc_use_laynorm_inp=str_to_bool(options.fc_use_laynorm_inp)
    fc_use_batchnorm_inp=str_to_bool(options.fc_use_batchnorm_inp)
    fc_use_batchnorm=list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
    fc_use_laynorm=list(map(str_to_bool, options.fc_use_laynorm.split(',')))
    fc_act=list(map(str, options.fc_act.split(',')))
    
    #[class]
    class_lay=list(map(int, options.class_lay.split(',')))
    class_drop=list(map(float, options.class_drop.split(',')))
    class_use_laynorm_inp=str_to_bool(options.class_use_laynorm_inp)
    class_use_batchnorm_inp=str_to_bool(options.class_use_batchnorm_inp)
    class_use_batchnorm=list(map(str_to_bool, options.class_use_batchnorm.split(',')))
    class_use_laynorm=list(map(str_to_bool, options.class_use_laynorm.split(',')))
    class_act=list(map(str, options.class_act.split(',')))
    
    
    wav_lst_te=ReadList(te_lst)
    snt_te=len(wav_lst_te)
    
    
    # Folder creation
    try:
        os.stat(output_folder)
    except:
        os.mkdir(output_folder)
    
    
    # loss function
    
    cost = nn.NLLLoss()
    
    
    # Converting context and shift in samples
    wlen=int(fs*cw_len/1000.00)
    wshift=int(fs*cw_shift/1000.00)
    
    # Batch_dev
    Batch_dev=128
    
    
    # Feature extractor CNN
    CNN_arch = {'input_dim': wlen,
              'fs': fs,
              'cnn_N_filt': cnn_N_filt,
              'cnn_len_filt': cnn_len_filt,
              'cnn_max_pool_len':cnn_max_pool_len,
              'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
              'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
              'cnn_use_laynorm':cnn_use_laynorm,
              'cnn_use_batchnorm':cnn_use_batchnorm,
              'cnn_act': cnn_act,
              'cnn_drop':cnn_drop,
              }
    
    CNN_net=CNN(CNN_arch)
    CNN_net.to(device)
    
    
    
    DNN1_arch = {'input_dim': CNN_net.out_dim,
              'fc_lay': fc_lay,
              'fc_drop': fc_drop,
              'fc_use_batchnorm': fc_use_batchnorm,
              'fc_use_laynorm': fc_use_laynorm,
              'fc_use_laynorm_inp': fc_use_laynorm_inp,
              'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
              'fc_act': fc_act,
              }
    
    DNN1_net=MLP(DNN1_arch)
    DNN1_net.to(device)
    
    
    DNN2_arch = {'input_dim':fc_lay[-1] ,
              'fc_lay': class_lay,
              'fc_drop': class_drop,
              'fc_use_batchnorm': class_use_batchnorm,
              'fc_use_laynorm': class_use_laynorm,
              'fc_use_laynorm_inp': class_use_laynorm_inp,
              'fc_use_batchnorm_inp':class_use_batchnorm_inp,
              'fc_act': class_act,
              }
    
    
    DNN2_net=MLP(DNN2_arch)
    DNN2_net.to(device)
    
    
    checkpoint_load = torch.load(model_file)
    CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
    DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
    DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])
    
    
    
    CNN_net.eval()
    DNN1_net.eval()
    DNN2_net.eval()
    test_flag=1
    
    
    d_vector_dim=fc_lay[-1]
    d_vect_dict={}
    
    
    with torch.no_grad():
    
        for i in range(snt_te):
    
             [signal, fs] = sf.read(data_folder+'/'+wav_lst_te[i])
    
             # Amplitude normalization
             signal=signal/np.max(np.abs(signal))
             #print("signal",signal.shape)
             signal=torch.from_numpy(signal).float().to(device).contiguous()
    
             if avoid_small_en_fr:
                 # computing energy on each frame:
                 beg_samp=0
                 end_samp=wlen
    
                 N_fr=int((signal.shape[0]-wlen)/(wshift))
                 Batch_dev=N_fr
                 en_arr=torch.zeros(N_fr).float().contiguous().to(device)
                 count_fr=0
                 count_fr_tot=0
                 while end_samp<signal.shape[0]:
                    en_arr[count_fr]=torch.sum(signal[beg_samp:end_samp].pow(2))
                    beg_samp=beg_samp+wshift
                    end_samp=beg_samp+wlen
                    count_fr=count_fr+1
                    count_fr_tot=count_fr_tot+1
                    if count_fr==N_fr:
                        break
    
                 en_arr_bin=en_arr>torch.mean(en_arr)*0.1
                 en_arr_bin.to(device)
                 n_vect_elem=torch.sum(en_arr_bin)
    
                 if n_vect_elem<10:
                     print('only few elements used to compute d-vectors')
                     sys.exit(0)
    
    
    
             # split signals into chunks
             beg_samp=0
             end_samp=wlen
    
             N_fr=int((signal.shape[0]-wlen)/(wshift))
    
             
             sig_arr=torch.zeros([Batch_dev,wlen]).float().to(device).contiguous()
             dvects=Variable(torch.zeros(N_fr,d_vector_dim).float().to(device).contiguous())
             count_fr=0
             count_fr_tot=0
             while end_samp<signal.shape[0]:
                 #print(sig_arr[count_fr,:].shape)
                 sig_arr[count_fr,:]=signal[beg_samp:end_samp]
                 
                 beg_samp=beg_samp+wshift
                 end_samp=beg_samp+wlen
                 count_fr=count_fr+1
                 count_fr_tot=count_fr_tot+1
                 
                 if count_fr==Batch_dev:
                     inp=Variable(sig_arr)
                     
                     #gc.collect()
                     #torch.cuda.empty_cache()
                     #torch.cuda.memory_summary(device=None, abbreviated=False)
                     #print("torch:",torch.cuda.memory_summary(device=None, abbreviated=False))
                     dvects[count_fr_tot-Batch_dev:count_fr_tot,:]=DNN1_net(CNN_net(inp))
                     count_fr=0
                     sig_arr=torch.zeros([Batch_dev,wlen]).float().to(device).contiguous()
    
             
             if count_fr>0:
              inp=Variable(sig_arr[0:count_fr])
              dvects[count_fr_tot-count_fr:count_fr_tot,:]=DNN1_net(CNN_net(inp))
    
             if avoid_small_en_fr:
                 dvects=dvects.index_select(0, (en_arr_bin==1).nonzero().view(-1))
    
             # averaging and normalizing all the d-vectors
             d_vect_out=torch.mean(dvects/dvects.norm(p=2, dim=1).view(-1,1),dim=0)
    
             # checks for nan
             nan_sum=torch.sum(torch.isnan(d_vect_out))
             
             if nan_sum>0:
                 #print(wav_lst_te[i])
                 sys.exit(0)
             #print("for sig before",torch.cuda.memory_allocated())
            # del sig_arr
            # torch.cuda.synchronize()
            # print("for sig after",torch.cuda.memory_allocated())
             #print("checking")
             # saving the d-vector in a numpy dictionary
             #dict_key=wav_lst_te[i].split('/')[-2]+'/'+wav_lst_te[i].split('/')[-1]
             dict_key=wav_lst_te[i]
             d_vect_dict[dict_key]=d_vect_out.cpu().numpy()
             
             if index==0:
                 print("if loop 0:",index) 
                 dict1=d_vect_dict
             if index==1:
                 print("if loop 1:",index) 
                 dict2=d_vect_dict
                
             #print("if",dict_key)
             #print("for dict before",torch.cuda.memory_allocated())
             #del dict_key
             #torch.cuda.synchronize()
             #print("for dict after",torch.cuda.memory_allocated())
# Save the dictionary
#print("checking")

#np.save(out_dict_file, d_vect_dict)

speaker1=list(dict1.values())


speaker2=list(dict2.values())
print("speaker1: ")
print(speaker1)
print("speaker2: ")
print(speaker2)
first_speaker=np.reshape(speaker1, (-1, 1)).T
second_speaker=np.reshape(speaker2, (-1, 1)).T
print("haha")
result_cos_sim=cosine_similarity(first_speaker,second_speaker)

print(result_cos_sim)
if result_cos_sim>=0.98:
    print(speakers[0],"and",speakers[1], "are the same")
elif result_cos_sim<0.98:
    print(speakers[0],"and",speakers[1], "are not the same")     
#print("dict: ",d_vect_dict)
