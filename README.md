# KDD-Cup2021-Multi-dataset-Time-Series-Anomaly-Detection
Solution in KDD Cup2021 Multi-dataset Time Series Anomaly Detection Competition

This repository is not the full code of the competition, I put the original code on google colab but my google account was stolen. While, my ranking was not very good anyway, so just show some thoughts on this anomaly detection competition.

In this competition, I used **weight average** and **EMD** method to preprocess data and **Spectral Residual**, **RRCF** and **AE/VAE** model to generate anomaly score.

Here is the leaderboard :-P
![leader_board](/images/utils/leaderboard.png)

## Data Show

Total datasets size is **250**, the files use a naming convention that provides a split between test and train: id_name_split-number.txt, Here split-number indicates there will be an anomaly from that position onwards.

Data can be split into 5 parts by the style of series, blue parts before the partition and red after:

#### Stationary series with jump-point anomaly
Series with suddenly and extremely high or low value beyond normal interval. 

<img src="images\diagnosis\origin_data\sj\10.png" width="225"/><img src="images\diagnosis\origin_data\sj\15.png" width="225"/><img src="images\diagnosis\origin_data\sj\36.png" width="225"/><img src="images\diagnosis\origin_data\sj\70.png" width="225"/>

#### Stationary series with no-responding anomaly
Series with suddenly 0 or low value while still in normal interval. 

<img src="images\diagnosis\origin_data\nr\17.png" width="225"/><img src="images\diagnosis\origin_data\nr\121.png" width="225"/><img src="images\diagnosis\origin_data\nr\149.png" width="225"/><img src="images\diagnosis\origin_data\nr\216.png" width="225"/>

#### Stationary series with period anomaly
Part of series have longer or shorter period, usually along with value shake.

<img src="images\diagnosis\origin_data\pr\3.png" width="225"/><img src="images\diagnosis\origin_data\pr\33.png" width="225"/><img src="images\diagnosis\origin_data\pr\127.png" width="225"/><img src="images\diagnosis\origin_data\pr\161.png" width="225"/>


#### Non-stationary series with unknown anomaly
Some series are non-stationary which are hard to distinguish anomaly artificially. Some smoothing preprocessing method can solve this kind of data.

<img src="images\diagnosis\origin_data\ns\4.png" width="225"/><img src="images\diagnosis\origin_data\ns\203.png" width="225"/><img src="images\diagnosis\origin_data\ns\225.png" width="225"/><img src="images\diagnosis\origin_data\ns\226.png" width="225"/>

#### Weried series
Some series are seem like more anomaly in blue parts, in these series, all model will select the position before partition which is not satisfied with rules. Therefore, I add a strategy that if the model show anomaly before the partition, then throw the blue part away and do once more selection.

<img src="images\diagnosis\origin_data\wd\42.png" width="225"/><img src="images\diagnosis\origin_data\wd\85.png" width="225"/><img src="images\diagnosis\origin_data\wd\205.png" width="225"/><img src="images\diagnosis\origin_data\wd\208.png" width="225"/>

## Preprocess

#### Weight average smoothing
Average inside a time window is typically a common preprocessing method. In my test, I get the difference between original data and averaged data to generate a new feature. This feature are mostly stationary and some can enlarge the anomaly. Here is the effect:

<img src="images\diagnosis\preprocess\aver\6.png" width="450"/><img src="images\diagnosis\preprocess\aver\11.png" width="450"/><img src="images\diagnosis\preprocess\aver\43.png" width="450"/><img src="images\diagnosis\preprocess\aver\201.png" width="450"/>

#### EMD
Empirical Mode Decomposition (EMD) is a method of decomposing signal into Intrinsic Mode Functions (IMFs) based on algorithm presented in [Huang et al](https://royalsocietypublishing.org/doi/abs/10.1098/rspa.1998.0193). This algorithm can deal with any signal without priority and the decomposed IMF can enrich the feature, but it is too slow. Empirically, the first few IMFs contain the main information, so it can be accelerated by reducing the number of decomposition.

<img src="images\diagnosis\preprocess\emd\10.png" width="450"/><img src="images\diagnosis\preprocess\emd\24.png" width="450"/>

## Model

After preprocess, I get mutivariate series (average-diff and IMFs). While in model, I still use univarite series to train and eval, then do scale and weighted sum of them to get the final submission.

#### Spectral Residual

I notice that most of series are like vibration signal, so frequency feature should have strong imformation to distinguish the anomaly. While, we should do fast fourier transformation to get series in frequency field and do reverse operation to reconstruct the original series. [Spectral Residual](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4270292&tag=1) is an effective means.

<img src="images\diagnosis\model\sr\24.png" width="450"/><img src="images\diagnosis\model\sr\53.png" width="455"/><img src="images\diagnosis\model\sr\10.png" width="450"/><img src="images\diagnosis\model\sr\44.png" width="450"/>

From the result of test, SR can solve period anomaly well, but fail to recognize other anomaly.

#### RRCF

[RRCF](https://github.com/kLabUM/rrcf/tree/master/rrcf) is an unsupervised anomaly detection model based on Isolation Forest. It used tree structure displacement to find anomaly and has shown great effect on suddenly changed situation.

<img src="images\diagnosis\model\rrcf\0.png" width="450"/><img src="images\diagnosis\model\rrcf\3.png" width="450"/><img src="images\diagnosis\model\rrcf\10.png" width="450"/><img src="images\diagnosis\model\rrcf\83.png" width="450"/>

RRCF has three main parameters: **nums_trees**, **shingle_size**, **tree_size** and **tree_size** is the most important one. If there are several positions' anomaly score are same, then should increase the tree_size.

<img src="images\diagnosis\model\rrcf\10 on 256 tree size.png" width="450"/><img src="images\diagnosis\model\rrcf\10 on 512 tree size.png" width="450"/><img src="images\diagnosis\model\rrcf\10 on 2048 tree size.png" width="450"/><img src="images\diagnosis\model\rrcf\10 on 4096 tree size.png" width="450"/>

#### AutoEncoder

AE is also a functional model in unsupervised anomaly detection task, just use reconstruction loss to find the position. In VAE, there is another metric: KL Divergence. While I tried MLP-AE(VAE), CNN-AE(VAE), RNN-AE(VAE) and Transformer-AE(VAE) and Transformer-AE shot the best score.

<img src="images\diagnosis\model\ae\15.png" width="450"/><img src="images\diagnosis\model\ae\70.png" width="450"/><img src="images\diagnosis\model\ae\113.png" width="450"/><img src="images\diagnosis\model\ae\215.png" width="450"/>


While in 70th data(left upper one), the anomaly is too large. AE is focus to reduce the mse of the whole dataset, so the model while fail in detecting very large anomaly point. Fortunately, RRCF is very professional in this regard.
