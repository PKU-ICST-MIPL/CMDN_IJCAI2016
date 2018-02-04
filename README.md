# Introduction
This is the source code of our IJCAI paper "Cross-media Shared Representation by Hierarchical Learning with Multiple Deep Networks", Please cite the following paper if you use our code.

Yuxin Peng, Xin Huang, and Jinwei Qi, "Cross-media Shared Representation by Hierarchical Learning with Multiple Deep Networks", 25th International Joint Conference on Artificial Intelligence (IJCAI), pp. 3846-3853 , New York City, New York, USA, Jul. 9-15, 2016.

# Usage
1.Environment

Set up deepnet as the instruction of deepnet-master/INSTALL.txt.
  
2.Data

cd to the deepnet-master/deepnet/examples/CMDN/feature dir.  
put the data with matlab format in this folder, and run mat2npy.py to convert matlab format to numpy format. Detailed data format please see in mat2npy.py.
  
3.Set

parameter 'size' and 'dimensions' in the following files need to be modified according to the data scale:  
-sae_img/data/wikipedia.pbtxt  
-sae_txt/data/wikipedia.pbtxt  
-multimodal_dbn/data/wikipedia.pbtxt  
where parameter 'size' means the number of data, parameter 'dimensions' means the dimension of data.  
  
4.Run

	$sh runall.sh

For more information, please refer to our [paper](http://www.icst.pku.edu.cn/mipl/tiki-download_file.php?fileId=314)

# Related work
If you are interested in cross-media retrieval, check our paper:

Yuxin Peng, Xin Huang, and Yunzhen Zhao, "An Overview of Cross-media Retrieval: Concepts, Methodologies, Benchmarks and Challenges", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2017.

Visit our [Benchmark Website](http://www.icst.pku.edu.cn/mipl/xmedia) and [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information.
