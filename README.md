# Generative Hyphae Growth Algorithm

### Creation and Inspiration

This algorithm was created by Shaan Appel with support from Shivam Parikh and Hatim Khan for CS 194-028 (Computational Design and Fabrication) during Spring 2018. Much of the inspiration for creating this algorithm is from the [Inconvergent Blog](http://inconvergent.net/generative/) by Anders Hoff.


### Results

The algorithm produces both stl files for printing to an `stlexport` folder. It saves Blender files with keyframes autoset for creating animations to the `blend` folder. Example animations can be seen here: 

[![Growth Simulation Video](https://img.youtube.com/vi/lye-AdrAYNw/0.jpg)](https://www.youtube.com/watch?v=lye-AdrAYNw)


### Instructions for running

##### Installing Dependencies:

Before running the program, you must install Blender, CGAL, and all python libraries (tested and run with Python 3.6).

Install all python library dependencies:

```
pip install -r requirements.txt
```

Install [Blender](https://www.blender.org/download/) and set up [command line usage](https://docs.blender.org/manual/en/dev/render/workflows/command_line.html) for your device.

Install [CGAL](https://www.cgal.org/download.html).


##### Running the program:

To run the full program:

```
python grow_hyphae.py
```

With Optional Arguments:

```
python grow_hyphae.py input_filename rand_seed num_samples
```

- `input_filename` Specifies stl file to run algorithm on. *WORKS BEST ON LOW POLY STL FILES*
- `rand_seed` Integer giving random seed for repeatability
- `num_samples` Specifies number of samples to connect with generative tree. More samples results in more complex model. Simulations were run with 1000 - 3000 samples.












