iris
====
[http://archive.ics.uci.edu/ml/datasets/Iris](http://archive.ics.uci.edu/ml/datasets/Iris)

realized by Logistic Regression and Gradient Descend  

learning rate: alpha = 0.1  
weight decay: lambda = 0.1  
iteration number: 100  

accruancy of test: 100.00%  

####data set  
training set: 60%  
cross validation set: 20%  
testing set: 20%  

####some important matrices  
m: number of sample  
n: number of feature  
k: number of class  

X: m * (n + 1) matrix  
y: m * k matrix  
theta: k * (n + 1) matrix  

----

####cost function  
<img src='http://latex.numberempire.com/render?J%28%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5B-y%5E%7B%28i%29%7D%5Clog%28h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29%29-%281-y%5E%7B%28i%29%7D%29%5Clog%281-h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29%5D%20%2B%20%5Cfrac%7B%5Clambda%7D%7B2m%7D%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5Ctheta_%7Bj%7D%5E%7B2%7D&sig=7cd9b36c62c2892849b9fdd75a4bab6b'
alt='J(\theta) = \frac{1}{m} \sum_{i=1}^{m}[-y^{(i)}\log(h_{\theta}(x^{(i)}))-(1-y^{(i)})\log(1-h_{\theta}(x^{(i)})] + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_{j}^{2}' />

####gradient  
<img src='http://latex.numberempire.com/render?%5Cfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%5Ctheta_0%7D%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%29x_%7Bj%7D%5E%7B%28i%29%7D%5C%20%2C%5C%20j%20%3D%200&sig=3823a59fc45d80c61b51f058b3697619'
alt='\frac{\partial J(\theta)}{\partial\theta_0}=\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}\ ,\ j = 0' />  
<img src='http://latex.numberempire.com/render?%5Cfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%5Ctheta_j%7D%3D%5Cleft%28%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28h_%7B%5Ctheta%7D%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%29x_%7Bj%7D%5E%7B%28i%29%7D%5Cright%29%2B%5Cfrac%7B%5Clambda%7D%7Bm%7D%5Ctheta_%7Bj%7D%5C%20%2C%5C%20j%20%5Cgeqslant%201&sig=5af78f21388ba83d755728199ab86f9b'
alt='\frac{\partial J(\theta)}{\partial\theta_j}=\left(\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}\right)+\frac{\lambda}{m}\theta_{j}\ ,\ j \geqslant 1' />


----

run make\_data.py to build data set  
run __main__.py to train and test
