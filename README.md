# HPC-Project
<h3>Parallel Implementation of Image Seam Carving Techniques on GPU</h3>

<b>
Karan Mehta </br>
Shyam Naren Kandala </br>
Anjaneya Srujan Narkedamalli </br>
</b>

<b>Introduction</b>:</br>
Seam carving is a method for image resizing without distorting the essential features of an image</br>
used for displaying an image on devices with heterogenous screen sizes. This is done by assigning each</br>
pixel an energy value (which signifies its relative importance of a pixel in an image), </br>
and identifying either horizontal or vertical seams of pixels with least energy sum for</br>
removal to reduce the image size.
</br></br>

<b>Proposed Work</b>:</br>
The aim of the project is to parallelize two different Seam Carving approaches and
analyze their performances on UCI HPC GPU cluster. All the approaches start by the computing
the energy function using ‘‘Gradient Magnitude’ which is embarrassingly parallel . Then they
identify the seams to be removed, parallely, as follows: <br>
1. Compute the minimum energy sum path using Dynamic Programming.<br>
2. Compute the minimum energy sum path using Greedy algorithm (Dijkstra’s) <br>
We will also implement a serial version of seam carving using dynamic programming to use it as
test oracle.
