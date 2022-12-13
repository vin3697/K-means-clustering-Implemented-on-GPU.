Opencl implementation of K-Means Clustering.

How to run?

1. Be sure that you have got proper gcc compiler and linker settings in the visual studio project.

2. Now, you must add src and lib  folder from this repository.

3. Create folder's named as "Results" where the clustering results will be stored and "data" where 2D data samples will be stored.

4. i) The dataset(A sets-A1) which is used for project is taken from this webpage http://cs.joensuu.fi/sipu/datasets/.
   ii) The dataset is two dimensional and has total 3000 samples which is stored as dataset.csv file in "data" folder.

5. Update the macros Num_Cluster, N_Samples and Dimension in the kmeans.cpp. And also, update samples and dimension in kmeans.cl (if dataset is changed).

6. Check correctness of all paths in kmeans.cpp (for dataset.csv, kmeans.cl and results respectively.)

7. Build and run the program and you will be able to observe the performance and result which is stored in .csv file in "Results" folder.
