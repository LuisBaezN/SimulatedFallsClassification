# Classification of Simulated Falls 

This repo shows how to use an artifitial neural network to classify falls. In order to accomplish this, we used the data set "Simulated Falls and Daily Living Activities Dataset" provided by Ahmet Turan Özdemir and Billur Barshan.

The relevant information about this data si listed below:

* [17 Volunteers × Avg. 5 repetitions × 36 Movements]
* [36 Movements including 20 Falls and 16 Daily Living Activities 5 Sensors each includes 3 axis Accelerometer, Gyroscope and Magnetometer.]
* [Attribute Information: Xsens MTw Motion Tracking Kit]

## Technologies

We used the following languages to build the solution:

* [R]
* [Python]
* [Tensorflow]

Further information about the versions is included in the environment file.

## Launch

To extract the raw data from the data set an algorithm was implemented in R. This algorithm just extract two classes, and clean the data. So, tu run this example, firs run the Fall_Deep file to extract the data, and then, run ann file to build and predict.

The architecture of the artifitial neural network is : 300/150/70/1, with a sigmoid function at the end of the net. 