Download Link: https://assignmentchef.com/product/solved-cs_ds541-homework1-deep-learning
<br>



<ol>

 <li><strong>Python and Numpy Warm-up Exercises </strong>This part of the homework is intended to help you review your linear algebra and learn (or refresh your understanding of) how to implement linear algebraic operations in Python using numpy (to which we refer in the code below as np). For each of the problems below, write a method (e.g., problem1) that returns the answer for the corresponding problem. Put all your methods in one file called homework1 WPIUSERNAME.py</li>

</ol>

(e.g., homework1 jrwhitehill.py). See the starter file homework1 template.py.

In all problems, you may assume that the the dimensions of the matrices and/or vectors are compatible for the requested mathematical operations. <strong>Note</strong>: Throughout the assignment, please use np.array, not np.matrix.

<ul>

 <li>Given matrices <strong>A </strong>and <strong>B</strong>, compute and return an expression for <strong>A </strong>+ <strong>B</strong>. <strong>[ 1 pts ]</strong></li>

</ul>

<em>Answer </em>(freebie!): While it is completely valid to use np.add(A, B), this is unnecessarily verbose; you really should make use of the “syntactic sugar” provided by Python’s/numpy’s operator overloading and just write: A + B. Similarly, you should use the more compact (and arguably more elegant) notation for the rest of the questions as well.

<ul>

 <li>Given matrices <strong>A</strong>, <strong>B</strong>, and <strong>C</strong>, compute and return <strong>AB </strong>− <strong>C </strong>(i.e., right-multiply matrix <strong>A </strong>by matrix <strong>B</strong>, and then subtract <strong>C</strong>). Use dot or dot. <strong>[ 2 pts ]</strong></li>

 <li>Given matrices <strong>A</strong>, <strong>B</strong>, and <strong>C</strong>, return <strong>A</strong>, where represents the element-wise (Hadamard) product and &gt; represents matrix transpose. In numpy, the element-wise product is obtained simply with *. <strong>[ 2 pts ]</strong></li>

 <li>Given column vectors <strong>x </strong>and <strong>y</strong>, compute the inner product of <strong>x </strong>and <strong>y </strong>(i.e., <strong>x</strong><sup>&gt;</sup><strong>y</strong>). <strong>[ 2 pts ]</strong></li>

 <li>Given matrix <strong>A</strong>, return a matrix with the same dimensions as <strong>A </strong>but that contains all zeros. Use zeros. <strong>[ 2 pts ]</strong></li>

 <li>Given square matrix <strong>A </strong>and column vector <strong>x</strong>, use linalg.solve to compute <strong>A</strong><sup>−1</sup><strong>x</strong>. Do <strong>not </strong>explicitly calculate the matrix inverse itself (e.g., np.linalg.inv, A ** -1) because this is numerically unstable (and yes, it can sometimes make a big difference!). <strong>[ 2 pts ]</strong></li>

 <li>Given square matrix <strong>A </strong>and row vector <strong>x</strong>, use linalg.solve to compute <strong>xA</strong><sup>−1</sup>. Hint: <strong>AB </strong>= (<strong>B</strong><sup>&gt;</sup><strong>A</strong><sup>&gt;</sup>)<sup>&gt;</sup>. <strong>[ 3 pts ]</strong></li>

 <li>Given square matrix <strong>A </strong>and (scalar) <em>α</em>, compute <strong>A </strong>+ <em>α</em><strong>I</strong>, where <strong>I </strong>is the identity matrix with the same dimensions as <strong>A</strong>. Use eye. <strong>[ 2 pts ]</strong></li>

 <li>Given matrix <strong>A </strong>and integers <em>i,j</em>, return the <em>j</em>th column of the <em>i</em>th row of <strong>A</strong>, i.e., <strong>A</strong><em><sub>ij</sub></em>. <strong>[ 1 pts ]</strong></li>

 <li>Given matrix <strong>A </strong>and integer <em>i</em>, return the sum of all the entries in the <em>i</em>th row <em>whose column index is even</em>, i.e., <sup>P</sup><em><sub>j</sub></em><sub>:<em>j </em>is even </sub><strong>A</strong><em><sub>ij</sub></em>. Do <strong>not </strong>use a loop, which in Python can be very slow. Instead use the sum function. <strong>[ 2 pts ]</strong></li>

 <li>Given matrix <strong>A </strong>and scalars <em>c,d</em>, compute the arithmetic mean over all entries of <em>A </em>that are between <em>c </em>and <em>d </em>(inclusive). In other words, if S = {(<em>i,j</em>) : <em>c </em>≤ <strong>A</strong><em><sub>ij </sub></em>≤ <em>d</em>}, then compute . Use nonzero along with np.mean. <strong>[ 3 pts ]</strong></li>

 <li>Given an (<em>n</em>×<em>n</em>) matrix <strong>A </strong>and integer <em>k</em>, return an (<em>n</em>×<em>k</em>) matrix containing the right-eigenvectors of <strong>A </strong>corresponding to the <em>k </em>largest eigenvalues of <strong>A</strong>. Use linalg.eig. <strong>[ 3 pts ]</strong></li>

 <li>Given a <em>n</em>-dimensional column vector <strong>x</strong>, an integer <em>k</em>, and positive scalars <em>m,s</em>, return an (<em>n</em>×<em>k</em>) matrix, each of whose columns is a sample from multidimensional Gaussian distribution N(<strong>x </strong>+ <em>m</em><strong>z</strong><em>,s</em><strong>I</strong>), where <strong>z </strong>is an <em>n</em>-dimensional column vector containing all ones and <strong>I </strong>is the identity matrix. Use either random.multivariate normal or np.random.randn. <strong>[ 3 pts ]</strong></li>

 <li>Given a matrix <strong>A </strong>with <em>n </em>rows, return a matrix that results from <strong>randomly permuting </strong>the rows (but not the columns) in <strong>A</strong>. <strong>[ 2 pts]</strong></li>

</ul>

1

<h1>2.    Linear Regression via Analytical Solution</h1>

(a) Train an age regressor that analyzes a (48 × 48 = 2304)-pixel grayscale face image and outputs a real number ˆ<em>y </em>that estimates how old the person is (in years). Your regressor should be implemented using linear regression. The training and testing data are available here:

<ul>

 <li>https://s3.amazonaws.com/jrwprojects/age_regression_Xtr.npy</li>

 <li>https://s3.amazonaws.com/jrwprojects/age_regression_ytr.npy</li>

 <li>https://s3.amazonaws.com/jrwprojects/age_regression_Xte.npy</li>

 <li>https://s3.amazonaws.com/jrwprojects/age_regression_yte.npy</li>

</ul>

To get started, see the train age regressor function in homework1 template.py.

<strong>Note</strong>: you must complete this problem using only linear algebraic operations in numpy – you may <strong>not </strong>use any off-the-shelf linear regression software, as that would defeat the purpose.

Compute the optimal weights <strong>w </strong>= (<em>w</em><sub>1</sub><em>,…,w</em><sub>2304</sub>) for a linear regression model by deriving the expression for the gradient of the cost function w.r.t. <strong>w </strong>and <em>b</em>, setting it to 0, and then solving. Do <strong>not </strong>solve using gradient descent. The cost function is

<em>n</em>

<em>i</em>=1

where ˆ<em>y </em>= <em>g</em>(<strong>x</strong>;<strong>w</strong>) = <strong>x</strong><sup>&gt;</sup><strong>w </strong>and <em>n </em>is the number of examples in the training set

D<sub>tr </sub>= {(<strong>x</strong><sup>(1)</sup><em>,y</em><sup>(1)</sup>)<em>,…,</em>(<strong>x</strong><sup>(<em>n</em>)</sup><em>,y</em><sup>(<em>n</em>)</sup>)}, each <strong>x</strong><sup>(<em>i</em>) </sup>∈ R<sup>2304 </sup>and each <em>y</em><sup>(<em>i</em>) </sup>∈ R. Note that this simple regression model does not include a bias term (which we will add later); correspondingly, please do <strong>not </strong>include one in your own implementation. After optimizing <strong>w </strong>only on the <strong>training set</strong>, compute and report the cost <em>f</em><sub>MSE </sub>on the training set D<sub>tr </sub>and (separately) on the testing set D<sub>te</sub>. <strong>[ 12 pts ]</strong>

<ol start="3">

 <li><strong>Proofs </strong>For the proofs, please create a PDF (which you can generate using LaTex, or, if you prefer, a scanned copy of your <strong>legible </strong>handwriting).</li>

</ol>

<ul>

 <li>Prove that<strong> a</strong></li>

</ul>

for any two <em>n</em>-dimensional column vectors <strong>x</strong><em>,</em><strong>a</strong>. Hint: differentiate w.r.t. each element of <strong>x</strong>, and then gather the partial derivatives into a column vector. <strong>[ 4 pts ]</strong>

<ul>

 <li>Prove that<strong>x</strong></li>

</ul>

for any <em>n</em>-dimensional column vector <strong>x </strong>and any <em>n </em>× <em>n </em>matrix <strong>A</strong>. <strong>[ 8 pts ]</strong>

<ul>

 <li>Based on the theorem above, prove that</li>

</ul>

<strong>Ax</strong>

for any <em>n</em>-dimensional column vector <strong>x </strong>and any symmetric <em>n </em>× <em>n </em>matrix <strong>A</strong>. <strong>[ 2 pts ]</strong>

<ul>

 <li>Based on the theorems above, prove that</li>

</ul>

∇<strong><sub>x</sub></strong>h(<strong>Ax </strong>+ <strong>b</strong>)<sup>&gt; </sup>(<strong>Ax </strong>+ <strong>b</strong>)<sup>i </sup>= 2<strong>A</strong><sup>&gt; </sup>(<strong>Ax </strong>+ <strong>b</strong>)

for any <em>n</em>-dimensional column vector <strong>x</strong>, any symmetric <em>n </em>× <em>n </em>matrix <strong>A</strong>, and any constant <em>n</em>dimensional column vector <strong>b</strong>. <strong>[ 4 pts ]</strong>

<strong>Submission</strong>: Create a Zip file containing both your Python and PDF files, and then submit on Canvas.

<strong>Teamwork</strong>: You may complete this homework assignment either individually or in teams up to 2 people.

2