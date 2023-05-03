Download Link: https://assignmentchef.com/product/solved-cs182-assignment-1
<br>
In this assignment you will practice writing backpropagation code, and training Neural Networks and Convolutional Neural Networks. The goals of this assignment are as follows:

<ul>

 <li>understand <strong>Neural Networks</strong> and how they are arranged in layered architectures</li>

 <li>understand and be able to implement (vectorized) <strong>backpropagation</strong></li>

 <li>implement various <strong>update rules</strong> used to optimize Neural Networks</li>

 <li>implement <strong>batch normalization</strong> for training deep networks</li>

 <li>implement <strong>dropout</strong> to regularize networks</li>

 <li>effectively <strong>cross-validate</strong> and find the best hyperparameters for Neural Network architecture</li>

 <li>understand the architecture of <strong>Convolutional Neural Networks</strong> and train gain experience with training these models on data</li>

</ul>

<h2><a id="user-content-setup" class="anchor" href="https://github.com/Dhanush123/cs182/tree/master/assignment1#setup" aria-hidden="true"></a>Setup</h2>

Make sure your machine is set up with the assignment dependencies.

<strong>[Option 1] Use Anaconda:</strong> The preferred approach for installing all the assignment dependencies is to use <a href="https://www.continuum.io/downloads" rel="nofollow">Anaconda</a>, which is a Python distribution that includes many of the most popular Python packages for science, math, engineering and data analysis. Once you install it you can skip all mentions of requirements and you are ready to go directly to working on the assignment.

<strong>[Option 2] Manual install, virtual environment:</strong> If you do not want to use Anaconda and want to go with a more manual and risky installation route you will likely want to create a <a href="http://docs.python-guide.org/en/latest/dev/virtualenvs/" rel="nofollow">virtual environment</a> for the project. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run the following:

<strong>Download data:</strong> Once you have the starter code, you will need to download the CIFAR-10 dataset. Run the following from the <code>assignment1</code> directory:

<strong>Compile the Cython extension:</strong> Convolutional Neural Networks require a very efficient implementation. We have implemented of the functionality using <a href="http://cython.org/" rel="nofollow">Cython</a>; you will need to compile the Cython extension before you can run the code. From the <code>deeplearning</code> directory, run the following command:

<strong>Start IPython:</strong> After you have the CIFAR-10 data, you should start the IPython notebook server from the <code>assignment1</code> directory. If you are unfamiliar with IPython, you should read our <a href="https://cs231n.github.io/ipython-tutorial/" rel="nofollow">IPython tutorial</a>.

<strong>NOTE:</strong> If you are working in a virtual environment on OSX, you may encounter errors with matplotlib due to the <a href="https://matplotlib.org/faq/virtualenv_faq.html" rel="nofollow">issues described here</a>. You can work around this issue by starting the IPython server using the <code>start_ipython_osx.sh</code> script from the <code>assignment1</code> directory; the script assumes that your virtual environment is named <code>.env</code>.

<h3><a id="user-content-submitting-your-work" class="anchor" href="https://github.com/Dhanush123/cs182/tree/master/assignment1#submitting-your-work" aria-hidden="true"></a>Submitting your work:</h3>

Once you are done working run the <code>collectSubmission.sh</code> script; this will produce a file called <code>assignment1.zip</code>. Upload this file to bCourses as per the assignment instructions.

<h3><a id="user-content-q1-fully-connected-neural-network-30-points" class="anchor" href="https://github.com/Dhanush123/cs182/tree/master/assignment1#q1-fully-connected-neural-network-30-points" aria-hidden="true"></a>Q1: Fully-connected Neural Network (30 points)</h3>

The IPython notebook <code>FullyConnectedNets.ipynb</code> will introduce you to our modular layer design, and then use those layers to implement fully-connected networks of arbitrary depth. To optimize these models you will implement several popular update rules.

<h3><a id="user-content-q2-batch-normalization-30-points" class="anchor" href="https://github.com/Dhanush123/cs182/tree/master/assignment1#q2-batch-normalization-30-points" aria-hidden="true"></a>Q2: Batch Normalization (30 points)</h3>

In the IPython notebook <code>BatchNormalization.ipynb</code> you will implement batch normalization, and use it to train deep fully-connected networks.

<h3><a id="user-content-q3-dropout-10-points" class="anchor" href="https://github.com/Dhanush123/cs182/tree/master/assignment1#q3-dropout-10-points" aria-hidden="true"></a>Q3: Dropout (10 points)</h3>

The IPython notebook <code>Dropout.ipynb</code> will help you implement Dropout and explore its effects on model generalization.

<h3><a id="user-content-q4-convnet-on-cifar-10-30-points" class="anchor" href="https://github.com/Dhanush123/cs182/tree/master/assignment1#q4-convnet-on-cifar-10-30-points" aria-hidden="true"></a>Q4: ConvNet on CIFAR-10 (30 points)</h3>

In the IPython Notebook <code>ConvolutionalNetworks.ipynb</code> you will implement several new layers that are commonly used in convolutional networks. You will train a (shallow) convolutional network on CIFAR-10, and it will then be up to you to train the best network that you can.

<h3><a id="user-content-q5-do-something-extra-up-to-10-points" class="anchor" href="https://github.com/Dhanush123/cs182/tree/master/assignment1#q5-do-something-extra-up-to-10-points" aria-hidden="true"></a>Q5: Do something extra! (up to +10 points)</h3>

In the process of training your network, you should feel free to implement anything that you want to get better performance. You can modify the solver, implement additional layers, use different types of regularization, use an ensemble of models, or anything else that comes to mind. If you implement these or other ideas not covered in the assignment then you will be awarded some bonus points.

5/5 - (4 votes)

<pre><span class="pl-c1">cd</span> assignment1sudo pip install virtualenv      <span class="pl-c"># This may already be installed</span>virtualenv .env                  <span class="pl-c"># Create a virtual environment</span><span class="pl-c1">source</span> .env/bin/activate         <span class="pl-c"># Activate the virtual environment</span>pip install -r requirements.txt  <span class="pl-c"># Install dependencies</span><span class="pl-c"># Work on the assignment for a while ...</span>deactivate                       <span class="pl-c"># Exit the virtual environment</span></pre>

<pre><span class="pl-c1">cd</span> deeplearning/datasets./get_datasets.sh</pre>