# DropBack
An emulation of the DropBack algorithm using Chainer.  https://arxiv.org/abs/1806.06949

## Using DropBack

If you are already familiar with Chainer, all you have to do is replace your updater with the DropBack updater, with tracked_size set the the value you want to test for your network. 

DropBack has only been tested with standard SGD, SGD + momentum or other optimizers might not work, since accumulated gradients are no longer scaled predictably by the learning rate.  

## Detailed guide TODO
