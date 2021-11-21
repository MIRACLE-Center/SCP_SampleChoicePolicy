
### preprocessing
1. max_sim_by_sift.py --tag xx
   maxsim_sift.py
   calculate all relations between samples (use all samples as template)
   对于每一张图：
   取出他的287个特征点，
   在其他的图上对于每个点找到最大的相似度
   得到一个 (n, n, 287) 的矩阵
   
2. test_by_multi.py (in ../cas-qs)
   randomly choose samples as templates , and get their MRE, MaxSim(landmarks), IDs
   对于选取的一些templates， 计算他们的MRE， Maxsim等

### Calc the details
1. select_ids.py --tag xx 
   根据 maxsim_sift 的结果，SIFT点，选择templates
   
2. test_specific_ids.py
   according to these choosen samples, get their MaxSim(SIFT), get draw maps, and so on.
   测试某个template/templates的性能