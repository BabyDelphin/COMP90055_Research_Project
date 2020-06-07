# COMP90055_Research_Project
## DeepFake Detection
This is the Git Repository for COMP90055 Research Project Team DeepFake Detection  
Team Members:  
Xingjian Zhang  SID: 1030767  
Deng Pan        SID: 354059  
Lixian Sun      SID: 938295  
Rui Wang        SID: 978296  

### Introduction
In  this  project,  we took  FaceForensics++ as  the source  video data,  and  trained two 
Neural Networks including  Xception  and  MobileNet with preprocessed images. The training
of each network produces five models, corresponding to four different dee-pfake technology
/software: Deepfakes,  Face2Face, FaceSwap,  NeuralTextures.  The  result of  the  model's 
evaluations with the test dataset has shown high precision.

### Data Source
In this project, FaceForensics++ was chosen as the dataset.
FaceForensics++ is a forensics dataset consisting  of  1000 original video  sequences that 
have been manipulated with four automated face manipulation methods: Deepfakes, Face2Face,
FaceSwap and NeuralTextures.
The github repository of FaceForensics++ can be found in the link below:  
https://github.com/ondyari/FaceForensics

### Video Processing and Face Cropping
Based on  the  selected  data  set,  the pre-processing module consists  of  three  steps: 
intercepting  frames  from video,  detecting faces from pictures, and saving face areas as 
new pictures.
![preprocess_flowchart]()

### Training and Evaluating
In consideration that pictures from the same video may have some internal  similarity,  we 
prefer that pictures extracted from the same videos  are in the same set. If the model has
seen some pictures from one video, the model  will  have  a  larger  chance  to  correctly 
predict other pictures from this video by catching some  features  of  the  video  itself.
However, what we want is catching those common features  among  different videos. Thus, we 
choose to split the dataset by video instead of by pictures. The average length of  videos 
are different from each  other. As a result,  when the video number of the training  set : 
the video number of the testing set is 80%:20%,  the ratio of picture numbers is  slightly 
different.
![split_dataset]()
Instead of presenting only an accuracy  for  the  binary  classification,  the  evaluation
matrices are based on true positive rate and  true  negative  rate  of  the  lassification 
result. The  definition of each classification  concept is  presented in  the table below.
The real video detection accuracy and fake video detection accuracy and overall  detection 
accuracy can be computed with the following equations respectively.
![evaluation]()

### Voting Mechanism
The trained model certainly is ready for image detection, but still a voting mechanism away
to do video prediction. How the voting mechanism works is  that once  we  have face  images 
from a video, these images will go through all four trained models. Every model has its own 
standard of predicting if a video is fake or not. For example, a video produces 300 images,
and Deepfakes’ threshold is that if over 50%  of images  are  predicted  as  fake, then the 
output  from  Deepfakes’  model is Fake. If any of  four models’  output is Fake, then this 
video is Fake.
![voting_1]()
![voting_2]()





