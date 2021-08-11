# Image Caption Generation using Spatial Attention

This is a Pytorch implementation of _Spatial Attention_ model proposed in _Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning_ published in CVPR 2017 [link](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lu_Knowing_When_to_CVPR_2017_paper.pdf)

#### Note: This is a work in progress. I will upload detailed explanation and results from MSCOCO dataset as well.

## Results
I have used Flickr8k and Flickr30k datasets for experiments. In the paper, authors train both the Caption Generator LSTM and the Encoder CNN (they use ResNet-152 CNN), i.e., they fine-tune the weights of pre-trained ResNet-152 CNN (which was originally trained for Object Recognition task). However, in most methods proposed in literature, the Encoder CNN has not been fine-tuned. Also, most earlier methods have used VGG-16 CNN. The choice of CNN influences Caption Generation performance, as noted in Katiyar et. al. [link](https://arxiv.org/abs/2102.11506). Hence, I have performed two sets of experiments: (a) Only Decoder (Caption-Generator) is trained with no fine-tuning of CNN. VGG-16 CNN is used as Encoder. (b) CNN is fine-tuned with learning rate of 1e-5 and Caption Generator is trained with learning rate of 4e-4. The authors of paper have trained their model for 80 epochs and started CNN fine-tuning after completion of first 20 epochs. However, due to resource constraints I have trained the model for 20 epochs only and I have trained both CNN and Decoder right from the beginning.

### Summary

The results on Flickr30k dataset can be summarized as:

|Method | CNN Fine-Tune |Beam | BLEU-1 | BLEU-2 | BLEU-3| BLEU-4| METEOR | CIDEr | SPICE | ROUGE-L |
|---|---|---|---|---|---|---|---|---|---|---|
| Original Implementation | Yes | 3 | 0.644 | 0.462 | 0.327 | 0.231 | 0.202 | 0.493 | 0.145 | 0.467 |



**For Flickr8k dataset:**

In the paper, the authors have not provided results on Flickr8k Dataset. Hence, comparative analysis is not possible.
The following table contains results obtained from our experiments without fine-tuning the CNN. VGG-16 CNN is used for image representation.

|Beam | BLEU-1 | BLEU-2 | BLEU-3| BLEU-4| METEOR | CIDEr | SPICE | ROUGE-L |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.603 | 0.412 | 0.270 | 0.175 | 0.202 | 0.463 | 0.134 | 0.447 |
| 3 | 0.623 | 0.432 | 0.292 | 0.195 | 0.200 | 0.493 | 0.141 | 0.449 |
| 5 | 0.622 | 0.433 | 0.293 | 0.197 | 0.196 | 0.497 | 0.141 | 0.450 |
| 10 | 0.613 | 0.424 | 0.286 | 0.191 | 0.190 | 0.485 | 0.135 | 0.445 |
| 15 | 0.608 | 0.419 | 0.281 | 0.188 | 0.187 | 0.476 | 0.133 | 0.442 |
| 20 | 0.601 | 0.413 | 0.276 | 0.183 | 0.185 | 0.464 | 0.132 | 0.439 |

The following table contains results obtained by fine-tuning the CNN with learning rate of 1e-5. I have used ResNet-152 CNN for image representation which is the same CNN used in the paper. However, I have trained both CNN and LSTM together from the start rather than training the CNN after 20 epochs have been completed.

|Beam | BLEU-1 | BLEU-2 | BLEU-3| BLEU-4| METEOR | CIDEr | SPICE | ROUGE-L |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.631 | 0.448 | 0.302 | 0.200 | 0.212 | 0.522 | 0.141 | 0.467 |
| 3 | 0.663 | 0.483 | 0.337 | 0.230 | 0.211 | 0.566 | 0.150 | 0.480 |
| 5 | 0.669 | 0.487 | 0.340 | 0.232 | 0.209 | 0.565 | 0.148 | 0.483 |
| 10 | 0.665 | 0.486 | 0.341 | 0.234 | 0.207 | 0.568 | 0.147 | 0.483 |
| 15 | 0.657 | 0.482 | 0.337 | 0.230 | 0.202 | 0.555 | 0.144 | 0.481 |
| 20 | 0.650 | 0.475 | 0.331 | 0.225 | 0.201 | 0.545 | 0.143 | 0.477 |


**For Flickr30k dataset:**

In the paper, the authors have provided results on Flickr30k Dataset. Hence, I have compared the results of my implementation with results in paper.

The following table contains results obtained from our experiments without fine-tuning the CNN. VGG-16 CNN is used for image representation. As mentioned earlier, authors have fine-tuned the CNN and used ResNet-152 CNN for image representation (which performs better than VGG-16 for Image Caption Generation, as noted in Katiyar et. al. [link](https://arxiv.org/abs/2102.11506)).

|Method |Beam | BLEU-1 | BLEU-2 | BLEU-3| BLEU-4| METEOR | CIDEr | SPICE | ROUGE-L |
|---|---|---|---|---|---|---|---|---|---|
| Paper | 3 | 0.644 | 0.462 | 0.327 | 0.231 | 0.202 | 0.493 | 0.145 | 0.467 |
| Ours | 1 | 0.603 | 0.418 | 0.284 | 0.193 | 0.179 | 0.387 | 0.121 | 0.430 |
| Ours | 3 | 0.626 | 0.434 | 0.299 | 0.204 | 0.177 | 0.397 | 0.121 | 0.429 |
| Ours | 5 | 0.631 | 0.438 | 0.303 | 0.207 | 0.174 | 0.403 | 0.119 | 0.429 |
| Ours | 10 | 0.624 | 0.431 | 0.295 | 0.200 | 0.170 | 0.397 | 0.115 | 0.426 |
| Ours | 15 | 0.618 | 0.426 | 0.291 | 0.195 | 0.167 | 0.389 | 0.113 | 0.423 |
| Ours | 20 | 0.609 | 0.418 | 0.284 | 0.191 | 0.165 | 0.377 | 0.110 | 0.420 |

The following table contains results obtained from our experiments with CNN fine-tuning. I have used ResNet-152 CNN for image representation and trained CNN with learning rate of 1e-5 along with decoder which is trained with learning rate of 4e-4. In the paper, authors have started training the CNN after the completion of 20 epochs and the whole model is trained till 80 epochs but I have trained both CNN and Decoder upto 20 epochs.

|Method |Beam | BLEU-1 | BLEU-2 | BLEU-3| BLEU-4| METEOR | CIDEr | SPICE | ROUGE-L |
|---|---|---|---|---|---|---|---|---|---|
| Paper | 3 | 0.644 | 0.462 | 0.327 | 0.231 | 0.202 | 0.493 | 0.145 | 0.467 |
| Ours | 1 | 0.641 | 0.452 | 0.309 | 0.212 | 0.189 | 0.445 | 0.129 | 0.449 |
| Ours | 3 | 0.665 | 0.478 | 0.335 | 0.234 | 0.192 | 0.489 | 0.137 | 0.460 |
| Ours | 5 | 0.660 | 0.475 | 0.331 | 0.230 | 0.189 | 0.483 | 0.135 | 0.458 |
| Ours | 10 | 0.649 | 0.469 | 0.328 | 0.227 | 0.184 | 0.468 | 0.129 | 0.453 |
| Ours | 15 | 0.634 | 0.460 | 0.322 | 0.223 | 0.181 | 0.466 | 0.126 | 0.449 |
| Ours | 20 | 0.626 | 0.452 | 0.314 | 0.215 | 0.179 | 0.453 | 0.124 | 0.447 |

