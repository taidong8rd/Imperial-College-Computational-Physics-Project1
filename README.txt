The Python scripts are best executed using Spyder. 
Please place the data.csv file in the same directory as the code files.

The script parabolic_univariate.py corresponds to Tasks 3.1 to 4.1, covering both the 1D parabolic minimizer and the 2D univariate method.

The other .py files correspond to Task 4.2 (2D Newton’s method) and Task 5 (3D Newton’s method), as indicated by their filenames.

Files without _sympy in their names use manually derived expressions for the required derivatives. 
These versions are faster to run and include comments for clarity. 
In contrast, files with the _sympy label use the sympy package to symbolically compute the derivatives. 
Although they implement the same logic and produce identical results, they run significantly slower. 
These versions are provided to demonstrate the author's effort in automating the derivation process.

Each file is structured using code cells that separate function definitions from implementation and plotting. 
Please run the cells from top to bottom in order.




