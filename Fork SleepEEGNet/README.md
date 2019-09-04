# Said at the forefront
Forked from SajadMo/SleepEEGNet(https://github.com/SajadMo/SleepEEGNet)
Based on SleepEEGNet, we tried to add skip connection, fractal network and dilated convolution in the CNN part.
Screenshots of all results are stored in the image folder, and the attention map is in the attention_maps folder.

原作者的代码由于版本问题在python2.7不能直接跑，我们做了部分修改。（2019.9.4）

## Recruitments
* Python 2.7
* tensorflow/tensorflow-gpu 1.10
* numpy(latest)
* scipy(latest)
* matplotlib(latest)
* scikit-learn(latest)
* matplotlib(latest)
* imblearn 0.4
* pandas(latest)
* mne 0.14

## Installation library
* install imblearn 0.4 for python 2.7
```
cd imblearn0.4(py27)\imbalanced-learn-0.4.X
pip install .
```
* install mne 0.14 for python 2.7
```
cd mne0.14(py27)
pip install .
```

## Dataset and Data Preparation
* To download SC subjects from the Sleep_EDF (2013) dataset, use the below script:

```
cd data_2013
chmod +x download_physionet.sh
./download_physionet.sh
```

* To download SC subjects from the Sleep_EDF (2018) dataset, use the below script:
```
cd data_2018
chmod +x download_physionet.sh
./download_physionet.sh
```

Use below scripts to extract sleep stages from the specific EEG channels of the Sleep_EDF (2013) dataset:

```
python prepare_physionet.py --data_dir data_2013 --output_dir data_2013/eeg_fpz_cz --select_ch 'EEG Fpz-Cz'
python prepare_physionet.py --data_dir data_2013 --output_dir data_2013/eeg_pz_oz --select_ch 'EEG Pz-Oz'
```

## Train

* Modify args settings in seq2seq_sleep_sleep-EDF.py for each dataset.
![](https://github.com/daidaiershidi/Brain-Cognition-Homework/blob/master/Fork%20SleepEEGNet/images/seq2seq_sleep.jpg)
* We only changed the CNN part in the above image.
* For example, run the below script to train SleepEEGNET(skipconnect) model with the 20-fold cross-validation using Fpz-Cz channel of the Sleep_EDF (2013) dataset:

![](https://github.com/daidaiershidi/Brain-Cognition-Homework/blob/master/Fork%20SleepEEGNet/images/CNN_skipconnect_epoch120.png)
```
python seq2seq_sleep_sleep-EDF-skipconnect.py --data_dir data_2013/eeg_fpz_cz --output_dir outputs_2013 --n_folds 20
```
* For example, run the below script to train SleepEEGNET(conv) model with the 20-fold cross-validation using Fpz-Cz channel of the Sleep_EDF (2013) dataset:

![](https://github.com/daidaiershidi/Brain-Cognition-Homework/blob/master/Fork%20SleepEEGNet/images/CNN_conv128_epoch50.png)
```
python seq2seq_sleep_sleep-EDF-conv.py --data_dir data_2013/eeg_fpz_cz --output_dir outputs_2013 --n_folds 20
```
* For example, run the below script to train SleepEEGNET(dilatedconv) model with the 20-fold cross-validation using Fpz-Cz channel of the Sleep_EDF (2013) dataset:

![](https://github.com/daidaiershidi/Brain-Cognition-Homework/blob/master/Fork%20SleepEEGNet/images/CNN_dilatedconv5_epoch30.png)
```
python seq2seq_sleep_sleep-EDF-dilatedconv.py --data_dir data_2013/eeg_fpz_cz --output_dir outputs_2013 --n_folds 20
```

## Results
* Run the below script to present the achieved results by SleepEEGNet(skipconnect) model for Fpz-Cz channel.
```
python summary.py --data_dir outputs_2013/outputs_eeg_fpz_cz(skipconnect)
```
* Run the below script to present the achieved results by SleepEEGNet(dilatedconv) model for Fpz-Cz channel.
```
python summary.py --data_dir outputs_2013/outputs_eeg_fpz_cz(dilatedconv)
```
* Run the below script to present the achieved results by SleepEEGNet(conv) model for Fpz-Cz channel.
```
python summary.py --data_dir outputs_2013/outputs_eeg_fpz_cz(conv)
```
![](https://github.com/daidaiershidi/Brain-Cognition-Homework/blob/master/Fork%20SleepEEGNet/images/results.png)
## Conclusion

We propose three automatic sleep stage annotations based on SleepEEGNet using a single channel EEG signal.Change in the CNN part. 
* Add skip connection to the original CNN part, this can be retained the more feature.
* Replace the CNN part with a simple fractal network. Through a combination of multiple different depth networks, the shallow layer provides a quicker answer and the deep layer provides a more accurate answer in depth.
* Added dilated convolution based on fractal networks to monitor the same convolution to a greater time.

We evaluated the proposed method on a single EEG channel (ie Fpz-Cz channel) from the Physionet Sleep-EDF datasets published in 2013.
You can see that we used fewer training epochs and relatively fewer parameters, and we got fairly good results.
In fact, due to limitations in computing resources,and we still have other models to run, and time is too late. we only used 5 channels of output in the dilated convolution. Although this greatly reduces the parameters, it also greatly affects the accuracy. Because other networks, including the original network, the output channels are 128 channels. I personally think that if you increase the number of training rounds or increase the amount of parameters, the model can also achieve a good result.

## Visualization
* Run the below script to visualize attention maps of a sequence input (EEG epochs) for Fpz-Cz channel.
```
python visualize.py --data_dir outputs_2013/outputs_eeg_fpz_cz(skipconnect)
python visualize.py --data_dir outputs_2013/outputs_eeg_fpz_cz(conv)
python visualize.py --data_dir outputs_2013/outputs_eeg_fpz_cz(dilatedconv)
```
