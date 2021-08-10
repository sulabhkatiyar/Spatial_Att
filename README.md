# spatial_att

### Results

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


