"""
This implementation is based on the following repository:
https://github.com/jik876/hifi-gan

The original code is licensed under the MIT License:

MIT License

Copyright (c) 2020 Jungil Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch

class FeatureLoss( torch.nn.Module ):
    def __init__(
            self,
            ):
        super().__init__()
        return
    
    def forward(
            self,
            inputs,
            targets,
            ):
        fmap_r = targets
        fmap_g = inputs
        
        loss = 0
        for dr, dg in zip( fmap_r, fmap_g ):
            for rl, gl in zip( dr, dg ):
                loss += torch.mean( torch.abs( rl - gl ) )

        return loss

class GeneratorLoss( torch.nn.Module ):
    def __init__(
            self,
            ):
        super().__init__()
        return
    
    def forward(
            self,
            inputs,
            ):
        disc_outputs = inputs
        
        loss = 0
        #gen_losses = []
        for dg in disc_outputs:
            l = torch.mean( (1-dg)**2 )
            #gen_losses.append( l )
            loss += l

        return loss

class DiscriminatorLoss( torch.nn.Module ):
    def __init__(
            self,
            ):
        super().__init__()
        return

    def forward(
            self,
            inputs,
            targets,
            ):

        disc_real_outputs = targets
        disc_generated_outputs = inputs
        
        loss = 0
        #r_losses = []
        #g_losses = []
        for dr, dg in zip( disc_real_outputs, disc_generated_outputs ):
            r_loss = torch.mean( (1-dr)**2 )
            g_loss = torch.mean( dg**2 )
            loss += (r_loss + g_loss)
            #r_losses.append( r_loss.item() )
            #g_losses.append( g_loss.item() )
        
        return loss