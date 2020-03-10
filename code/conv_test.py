import torch
import torch.nn as nn
from tessst import Conv

class ConvTest():
	def test(self):
		X = torch.Tensor([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]]).view((1,1,5,5));
		K = torch.Tensor([[[[1,0,1],[0,1,0],[1,0,1]]]])
        
        	# manual implementation
		conv = Conv(K,3,stride=1,padding=1)
		manual = conv.forward(X)
		print(manual)

		#pytorch automatic implementation
		pytorch = nn.functional.conv2d(X, K, padding=1,stride=1)
		print(pytorch)
		return(torch.eq(manual,pytorch))
		
		
a = ConvTest()
a.test()



