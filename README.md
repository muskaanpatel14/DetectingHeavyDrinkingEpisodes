# DetectingHeavyDrinkingEpisodes
Final Project under CS2470 Deep Learning Course (Brown University - Master's in Computer Science)

Detecting Heavy Drinking Episodes using Accelerometer Data
Amrit Singh Rana, Muskaan Patel, Sarah Prakriti Peters

Introduction:
Excessive alcohol consumption is a significant health risk that accounted for 5.3% of deaths worldwide in 2016. Heavy alcohol use is particularly prevalent on college campuses, and social workers have investigated various interventions to reduce drinking habits, including mobile-based interventions. However, measuring blood alcohol content or transdermal alcohol content in real-time can be challenging. To address this issue, this project proposes a smartphone-based system that uses accelerometer signals to track a user's level of intoxication in real-time, enabling the delivery of just-in-time adaptive interventions during heavy drinking events. The system includes a heavy drinking classifier that makes classifications on windows of accelerometer data using deep learning techniques. The team will focus on developing novel features to improve the accuracy and usability of the classifier. This project builds on existing research and aims to contribute to the development of effective mobile interventions to reduce heavy drinking.

Methodology:
Data Pre-Processing: 
Binarized TAC Data: The first data pre-processing step involved binarizing the transdermal alcohol content (TAC) data, which was collected using a wristband sensor, into two categories: above and below the legal limit for driving. The legal limit for driving in the US is 0.08% TAC level. TAC values above this threshold were labeled as 1, indicating the participant was likely intoxicated, while values below this threshold were labeled as 0, indicating the participant was likely sober.
Upsampled TAC Data: The TAC data was collected at a lower frequency (one sample every 30 minutes) than the accelerometer data (~50 samples every millisecond), resulting in unevenly spaced observations. To address this, the TAC data was upsampled to match the frequency of the accelerometer data. 
Merged TAC Data with Accelerometer Data: The upsampled TAC data was then merged with the accelerometer data to create a single dataset. The accelerometer data was collected using a smartphone app, and included three-axis accelerometer readings.
Evenly Sampled Accelerometer Data: The accelerometer data was sampled at an uneven frequency (1 to 88 samples per millisecond). To create evenly sampled data, we sampled 20 observations from each window, while ignoring intervals with less than 20 observations. This 
Created Sliding Windows: Overlapping sliding windows of 10 milliseconds were created to generate independent data samples as an alternate data set. Think of each record as a mini time-series of 10 milliseconds each, having 200 x 3 observations (20 observations per millisecond, for 10 milliseconds, for x, y and z axes of the accelerometer). This sliding window approach enabled the researchers to capture changes in the participant's TAC levels over time and use this information to classify the level of intoxication.


Feature Engineering:
The first step in the feature engineering part was to convert the time series data into a windowed format. This was done by dividing the data into windows of 5 seconds, which was found to be an appropriate window size to capture the motion correctly. This process generated new features by aggregating the 100 raw signals within each window.
In stage 1 of the feature engineering process, 16 simple statistical features were built. These features included mean, standard deviation, min-max values, skew, kurtosis, and other basic statistics. These features were designed to capture the overall characteristics of the data.
In stage 2, 13 new features were added based on the fast Fourier transform (FFT). The FFT is a mathematical technique that transforms a time domain signal into a frequency domain. This technique was used to produce a new set of features that describe the frequency distribution of the signal. The new features included FFT Mean, FFT standard deviation, FFT Interquartile range, and other similar statistics.
In stage 3, the index values of the underlying data were used as potential features. This step was taken to capture any additional information that could be extracted from the data.
After applying all these feature engineering steps, the data frame now had 107 columns of data to work with. This rich set of features was designed to capture as much information as possible from the original time series data, allowing for the development of accurate classification algorithms.
Deep Learning Pipeline Modelling:
To build our deep learning models, we used three datasets: Vanilla Time-Series, Time-Series with Features, and Sliding Window Dataset. We ran three models: CNN, MLP, and LSTM on each of these datasets. 
Convolutional Neural Network: We chose a CNN with 1-D convolutions for its ability to extract features from time-series data. The three-axis accelerometer data can be treated as RGB color channels, and the CNN can learn patterns and features from these channels.  However, CNN performed poorly with an accuracy of only 31%, at best. 
Multi-Layered Perceptron: After receiving these results, we switched to a simpler network for our baseline model. We used an MLP, which is a type of feedforward neural network commonly used for classification tasks. Unlike CNN, it requires feature extraction to be performed before feeding the data to the network. The MLP model achieved an accuracy of approximately 68% in classifying the transformed accelerometer data for detecting heavy drinking episodes.
Long Short-Term Memory Networks (LSTM): LSTM is a type of recurrent neural network that can process time-series data by capturing long-term dependencies in sequential data. In this project, the LSTM model was found to be effective in analyzing accelerometer data and achieved a high accuracy of approximately 71% in classifying the data into two categories. The bidirectional aspect of the LSTM allows it to process the data in both forward and backward directions, making it especially adept at capturing temporal patterns and dependencies.
Results: The team pre-processed the data by binarizing, upsampling, merging, and evenly sampling it, followed by windowing techniques for feature engineering. The resulting dataset had 107 columns to work with. Three models, namely CNN, MLP, and LSTM, were applied to three datasets, namely Vanilla Time-Series, Time-Series with Features, and Sliding Window Dataset. Although CNN is known to be suitable for accelerometer data, it performed poorly, and the MLP model achieved an accuracy of approximately 68%. However, the LSTM model was highly effective in analyzing accelerometer data and achieved a high accuracy of approximately 71%. These results highlight the suitability of the LSTM approach for this type of data and suggest that it can be used to address the problem of excessive alcohol consumption effectively.
Challenges: One of the initial challenges we faced was to understand the dataset that we downloaded from UCI. Labels and Accelerometer data had different sampling rates. TAC data has a reading every 30 minutes while Accelerometer data has multiple samples every millisecond. We had to up-sample TAC data to match Accelerometer sampling frequency. We also noticed that each timestamp had between 1 to 88 records. We dropped all timestamps with less than 20 records, and randomly sampled 20 records from the remaining rows. While creating sliding windows, we played around with various sizes of 10,15,20 seconds. However, our models were not learning with sliding window data which is why we focussed on vanilla time series and feature engineering datasets. We faced another challenge with our baseline CNN model, which performed the worst. We did not notice any improvement in accuracy after adding multiple Conv1D layers. We had hoped to achieve good results with the CNN model due to feature extraction. 


Reflection: Reflecting on the project, we learned some valuable lessons. Firstly, we realized the importance of choosing good quality datasets with adequate research paper publications. The dataset we used for this project was challenging due to its large size, lack of proper documentation, and inadequate research publications. Secondly, we learned that preprocessing time-series data with 14 million instances is not an easy task, and it requires a lot of resources and time. We also learned that simple models like MLP can outperform more complex models, and it's essential not to underestimate their power. Additionally, we realized that our label data is laggy with respect to input variables and that we require more domain knowledge to handle TAC data.