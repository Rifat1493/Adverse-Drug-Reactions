# Adverse-Drug-Reactions
Increased usage and under reporting of adverse drug reactions (ADRs) of opioids instigates us to explore some other data sources like Twitter and PubMed. Our paper aims at discovering illegal trafficking of opioids as well as distinguishing tweets from having ADRs or not using binary classifier. We also evaluated the performance of MetaMap in finding ADRs from Twitter and compared the MedDRA encoding system on ADR terms found from tweets and PubMed. We used Latent Dirichlet Allocation (LDA) to find tweets related to illicit sale and used several neural networks for binary classification. It was reported that out of 98 ADRs found from tweets, 50 could be mapped to Lowest Level Terms (LLTs) and 48 to (Preferred Terms) PTs where only 23 LLTs and 15 PTs were reported from PubMed. Among the binary classifier Convolutional Recurrent Neural Network (CRNN) were found to be more promising with .71 F1 score though other models are close to the best one with little margin. Effect of skewness was also monitored in our study. Social media is a good choice for mining pharmacovigilance but during extraction a lot more noise data may come which needs to be avoided.
![](image.png)

If You find this repository as useful. Please cite:
Md Jamiur Rahman Rifat,Sheak Rashed Haider Noori, and Md Rashedul Hasan. ”Pharmacovigilance study of opioid drugs on Twitter and PubMed using artificial intelligence.” In 2019 Tenth International Conference on Computing, Communications and Networking Technologies (ICCCNT),IEEE.
