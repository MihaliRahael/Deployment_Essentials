# PRODUCTIONIZATION OF ML MODELS USING AWS, Docker, Github Actions, FastAPI, Heroku

## Content

-   Data stage
-   Training stage
-   Deployment stage
-   Steps for productionizing
-   Github Operations
-   Visual Studio Code operations
-   AWS Operations
-   Dockerize the entire application
-   Github Actions Operations
-   Configure EC2 as self-host runner
-   Actions YAML file explaination
-   Test the deployment
-   Deploy in Heroku cloud

Suppose we have done done EDA, feature engineering, experiments and trained some ML models. After model evaluation we need to deploy it in real world (ie productionization). We need to understand the challenges, options and design choices in deploying.

## Data stage

Raw data – csv/ text file/ DB/ Data warehouse/ Hive (specialized DB)/ Hadoop (Distributed storage)/ Spark storage. If we know SQL we would be able obtain data from the above storages (Hive SQL, Spark SQL etc.)

ETL stage (Extract transform and load) – Data processing pipeline. Data engineers use SQL and any programming lang to develop featurized data from raw data. One challenge is frequency of ETL pipeline. (Eg: Sales data – every hour there will be data. We might need to fetch and process this data every hour or every day). During the scenarios like this, if we need model retraining everyday, we need to run ETL pipeline on daily basis. Another design challenge is the size of data. Assume an internet company which handles tera bytes of data, we need to run ETL on hourly basis. (Note : TB range data is stored in Spark, Hadoop, DW etc.)

## Training stage

Once featurized data is ready, we need to start modelling (say LR here). One design challenge is volume of data. Check whether the model’s training performance if the volume increases as the below graph. Here we can see that above 300k points there is no considerable improvement in performance. In that case sample properly and train on 300K points.

![image](https://user-images.githubusercontent.com/106816732/213475004-4796d507-9fc7-41de-833b-b08f2f4a5202.png)

But if performance is increasing as points increase we have 3 options. 1) if we have small amount of data which may fit in RAM (32GB box) we can use sklearn or Xgboost 2) if data is of medium size which fits in HDD, we might need to do some sampling (which may decrease the performance as well). We can run the model in RAM itself. For example, divide the data into 3 parts. Each part may fit in RAM. Train LR1, LR2,LR3 and at the end take average or use any aggregation of all models to get the final one. This approach works mostly for simple algos like LR. 3) Real challenge if data is large like 10M/100M/1B points which my require 1TB/10TB or more. Several methods are there but common approach is use Spark ML. Spark is distributed computing platform. Ie data will be distributed among a set of discrete boxes having individual CPU,RAM,HDD and connected through a network and interbox data transfer is possible.

Imp link : <https://spark.apache.org/docs/latest/ml-guide.html>

**MLlib** is Spark’s machine learning (ML) library. Its goal is to make practical machine learning scalable and easy. At a high level, it provides tools such as:

-   ML Algorithms: common learning algorithms such as classification, regression, clustering, and collaborative filtering
-   Featurization: feature extraction, transformation, dimensionality reduction, and selection
-   Pipelines: tools for constructing, evaluating, and tuning ML Pipelines
-   Persistence: saving and load algorithms, models, and Pipelines
-   Utilities: linear algebra, statistics, data handling, etc.

If the dataset is so huge, we can write code using SparkML (ie we can use any algos) which is very easy. Refer the doc link.

## Deployment stage

Before going into deploy stage we need to understand the basic software architecture.

The end user enter the web url and expects a webpage to appear. Say he browsed for amazon.com. Request will reach amazon server. The server itself would be connected to various instances where boxes could be in Java, C++, Perl etc. The web page contains different parts like images, information, banking info , similar products etc. Each part of the page would call different service to fetch the data. This web server will pull all the required data from different services and collate and builds the final webpage and sent it to the user. In this particular example, similar products feature requires a ML service. This ML service could be residing in a completely different box, where it contains a model. So in order to fill up the blank space in web page, web server sent a request to this model by giving some information like userid, product details etc (which will be the input features and data to the model). Model predicts the list of similar products and returns to server through a recommendation engine or whatever model. So this whole thing is called the **SOA** (Service Oriented Architecture). (There are also some extensions called micro-services etc.) So in a nutshell, the ML team has a ML server where the model is running. We just need to make our model available using web api so that anybody can sent a ‘request’ with some data and the model will sent back ‘response’. We can modify the model without impacting the rest of the systems. As long as the webserver and ML server agree upon the format of the i/p and o/p data we can do anything in the back. This is generally called a **web API.** Very common format for request and response data is JSON.

Suppose we already built our model, so being able to call the ‘predict’ method on it shouldn’t involve building it again. It’s a lot of overhead and unnecessary. But how can we avoid having to recreate the model in our API? This is where data persistence in Python comes into play. The main goal of our API is to enable a client, whether a website or mobile app, to be able to use our model to make predictions. Before we can start building our API, we need a way to persist our model to a file so we can use that instead of training it every request. Python has a built-in method of persisting data called pickle. The pickle module can serialize objects or data into a file that we can save and load from.

The model in ML server could be in multiple form like persistent python object (which means we can store it in disk) or suppose if we use logistic regression model we could obtain the parameters of the model (w-coefficients, b-intercept) which can also be stored in to a disk or file. This is the model file which will be stored in the HDD of ML server.

## Steps for productionizing

### Github Operations

1.  Create Github profile
2.  Create a new repository in public mode
3.  Give repository name
4.  Check “Add a README file”
5.  Select gitignore template as Python. (gitignore contains the files and data which needn’t to get committed or push to github.
6.  Choose licence as Apache Licence 2.0
7.  We need to commit in this repo. So clone this repo into the local so that we will be able to commit on to it. In order to clone this repo
8.  Create a folder in local. Say we have created a folder named ‘ML End to end Project’ in D: drive.
9.  Goto CMD. Goto the path of folder using the commands. ‘D:’ ‘cd D:\\ML End to End Projects’
10. Goto github repo Code Local Copy the URL
11. Goto CMD Use command ‘git clone \<copied URL\>’
12. Copy the ipynb and pkl file we created to this cloned repo

### Visual Studio Code operations

1.  Open File Open Folder and upload the project folder to the VS Code explorer
2.  README.md (Sample format shown below)

![image](https://user-images.githubusercontent.com/106816732/213475766-71eab705-7220-46a5-bc71-2af99c683e5d.png)

3.  Creating an environment for the whole project.
*  New terminal Select command prompt “conda create -p venv python==3.7 -y”
*  Make sure the workspace folder is under the new environment.
*  Activate the environment using the command “conda activate venv/” and deactivate using “conda deactivate”
*  We can manually set the ipynb file to venv environment from the upper right portion. (Select venv (Python 3.7.0)
4.  Create requirements.txt file under the folder we are working. Define the libraries we require for the project. Run the command “pip install -r requirements.txt” in the terminal
5.  Configure Git CLI and git add/commit/push. This step is necessary in order to push/commit our code to the repo
6.  Create **setup.py** file (install everything in local environment) and edit the information. There are mainly two uses for this file. Sometimes even if we import the library, path defined is correct we might get error saying ‘module not found’. To avoid that we need to setup our project in such a way that our code can be updated or may be utilized as library. Things written inside setup.py will help to avoid this error.
7.  We define Fast API (or Flask API) inside **main.py**. Once a model is created, that model will be like a file, precompiled file (pkl file). This file will be deployed in cloud and an API will be created to communicate with that model. When an API is created we just need to pass the input, model will give the o/p

**Error Note :** Whenever we run a git command in CMD (windows or in visual studio code), if we face an error saying “‘git’ is not recognized as an internal or external command”, follow the steps.

1.  In the Start Menu or taskbar search, search for "environment variable".
2.  Select "Edit the system environment variables".
3.  Click the "Environment Variables" button at the bottom.
4.  Double-click the "Path" entry under "System variables".
5.  With the "New" button in the PATH editor, add C:\\Program Files\\Git\\bin\\ and C:\\Program Files\\Git\\cmd\\ to the end of the list.
6.  Close and re-open your console.

### AWS Operations

-   Login to AWS
-   Create an IAM user (we need to create a user using which we will deploy using git hub actions) with the following specific access
-   EC2 – where we deploy our project
-   S3 bucket – to store artifact and model
-   ECR (Elastic container registry) – To save the docker image
-   IAM dashboard Access management Users Add Users Give any user-name
-   Give access key – Programmatic access. If we don’t want the new user to login to the AWS console itself, make sure to uncheck the password. We can choose Access Key – programmatic access
-   Next page we set the permissions. When we are creating new user for the first time, we cant “copy permissions from existing user” or “Add user to group”. We can “attach existing policies directly”
-   In filter policies, search s3 and then opt “Amazon S3 full access”. Then search for “registry” and choose “AmazonEC2ContainerRegistryFullAccess”. Search ec2 and choose “AmazonEC2Full Access”. These are the policies. Note the exact name of it for using in the source code.
-   Go to tag section in the next page, which is completely optional. On the next page check the summary of what we created.
-   In the last stage, we should download

**Create S3 bucket**

-   We will create a bucket in ap-south-1
-   Go to S3 services Create bucket
-   Bucket name is globally unique. Select the region as above mentioned.
-   Rest of the things keep it as default and then create bucket

**Create ECR repo**

-   Go to ECR services Create repository
-   We can choose visibility setting as private (even public is fine)
-   Give repo name and then go for ‘create repository’
-   Copy the URL respective to the newly created ECR repo. Eg: 566712891.dkr.ecr.ap-south-1.amazonaws.com/sensor-fault

**Create an EC2 machine**

-   Go to EC2 service Launch instance
-   Give a name (sensor-vm for example). Choose Ubuntu server
-   Select t2.small as instance type
-   Generate key pair for SSH. Give key pair name and continue. Files will be downloaded
-   Network settings – Make sure “Create security group” is checked. Allow SSH traffic from ‘Anywhere’ and Allow HTTP and HTTPS traffic from the internet (since we have created a webapi)
-   Configure Storage – make 16 Gb
-   Launch Instance
-   Select the instance Connect Click ‘connect’ on ‘EC2 Instance Connect’ tab AWS CLI will open in browser itself (we won’t be using putty for connecting since we have very small work to do.)

Now we need to do some set up in EC2 Ubuntu machine.

-   Install Docker – commands as follows

    ![image](https://user-images.githubusercontent.com/106816732/213477129-1545e7c6-ca21-4d70-8bfb-abd1a6d58fa3.png)

Whenever we use docker commands we would use sudo. Last two commands is in order to avoid that repetitive usage.

### Dockerize the application

We need to make sure to we dockerize the entire application.

-   Create a new file name ‘Dockerfile’ in the environment. (Make sure using same naming convention as VS code will automatically detects the type of file)

Docker commands:

**FROM** python 3.7– use to select any kind of base image. From the docker hub (all the images are present over here), it will take the particular base image which has linux and on top of it python 3.7. Each image contains several sub layers.

**COPY** . /app – Copy all the files (ML application files we created till now) in the current local location to the location (folder) which I have named as ‘app’ in base image ie copying to the ‘app’ folder inside the docker image. We can change the name to anything.

**WORKDIR** /app - setting this app folder in base image as the working directory

**RUN** – install the requirements

**EXPOSE** \$PORT– When the docker image runs as container, in order to access the application inside it, we have to expose some port. When we are deploying in cloud, the server will automatically assign this particular port.

**CMD** **gunicorn --workers=4 --bind 0.0.0.0:\$PORT app:app** – This command is used to run my webapp or our entire application. Gunicorn actually helps to run this entire python web application inside the Heroku cloud. Gunicorn is required whenever we try to deploy anything in Heroku platform. Whenever a request is coming to the application workers will divide based on the instances. Say 1000 requests are coming, workers will divide it to four 250 processes to make it easy. 0.0.0.0 IP address will be the local address in the Heroku cloud.

OR

**CMD [“python”,”app.py”] -** This is basically ‘python app.py’ command in linux.

### Github Actions Operations

Github action is nothing but CICD pipeline ie as soon as I commit something to github, automatically deployment will happen in server. We will use Github Actions for doing continuous Integration continuous delivery and continuous deployment. High level overview of steps for deployment are as follows:

-   Build docker image of the source code
-   Push the image into ECR
-   Launch the EC2
-   Pull the image from ECR in EC2
-   Launch the docker image in EC2

### Configure EC2 as self-host runner

**What is a self-hosted runner?**

In GitHub, the runner is the application that runs a job from a GitHub Actions workflow. The runner can run on the hosted machine pools or on self-hosted environments. A project may want to use a self-hosted runner which, according to GitHub, offers more control of hardware, operating system, and software tools than GitHub-hosted runners provide. With self-hosted runners, you can choose to create a custom hardware configuration with more processing power or memory to run larger jobs, install software available on your local network, and choose an operating system not offered by GitHub-hosted runners. Self-hosted runners can be physical, virtual, container, on-premises, or in a cloud.

Procedure

-   Open the github repo we created Settings Actions Runners New self-hosted runner
-   Select Runner image as Linux. We can see the commands which needs to run in server to make EC2 instance a self host runner. In between while asking for entering the name of runner group, just press enter for default and for name of runner, use self-hosted (as shown below).

![image](https://user-images.githubusercontent.com/106816732/213477446-bbdec5d9-3b11-41c7-a6b9-e63c0b1e6f7d.png)

After this step follow the commands. To activate the runner we need to run ./run.sh

![image](https://user-images.githubusercontent.com/106816732/213477488-63bd2c29-b9e9-4f57-89ba-3ea703e688c3.png)

**Note** : We need to make sure self-hosted machine is available and up before deployment. Then only the deployment job inside github actions yaml will run. Settings Actions Runners Runners list should contain self-hosted and should be in green state (idle). If we break the AWS CLI in browser and close all the AWS pages, we can see self-hosted in offline mode.

Again connect EC2 using browser CLI, ls cd action-runner ./run.sh

Now the deployment job will run.

**Set up the environment variables in Github Actions**

Repo Settings Security Secrets Actions New repo secret

-   AWS_ACCESS_KEY_ID
-   AWS_SECRET_ACCESS_KEY
-   AWS_REGION
-   AWS_ECR_LOGIN_URI : 566712891.dkr.ecr.ap-south-1.amazonaws.com

Note : Don’t use ECR repo name in Login URL

-   ECR_REPOSITORY_NAME : sensor-fault
-   MONGO_DB_URL

### Github Actions main.yaml – explanation

**on**: whenever we make any push on main branch, the workflow will trigger except modifications in README file.

**continuous integration job** : will do the checkout (means there will be an ubuntu machine provided by git hub, which is a VM provided by github action, where all the source code we created will be copied here, that is checkout), linting code (checking the quality of code, like syntax, spacing, variables etc.) and run the unit tests (we haven’t written any code for unit testing, but it is required for several projects.)

**continuous delivery job:**

-   will do the checkout
-   running utility commands
-   configure aws credentials
-   Login to ECR
-   Build docker image of the source code we created
-   Push image to ECR

Note: after this stage we will be able to see the image created in the ECR

**continuous deployment job:**

Note : Commands in this portions will be running in our ec2 instance ie self-hosted we setup earlier.

-   configure aws credentials
-   pull docker image from ECR
-   Stop and remove the already running containers if any. If there is no containers then resume the process.

    Note: If we get any error at this stage of deployment, comment out this particular job and try.

    Note: We can go to EC2 CLI and check docker images and then run the command to stop (shown in main.yaml) to manually stop the containers running.

-   Launch the docker container
-   Clean the unnecessary images and containers ie whenever we create a new image, this command will clean up the old images. (Note the 3rd command will stop the container, not clean up.)

## Test the Deployment

1.  Go to EC2 instance and check **docker ps** for the container status
-   docker exec -it \<container id\> bash
-   ls : will show all the files and folders of our source code which got created inside the image
-   cd logs : will show all the logs
1.  Check S3 bucket whether the code has synced.
2.  Go to EC2 instance in UI. Check the Public IPV4 DNS in browser (check with http) (Don’t do the training using API)
3.  Inorder to test the model or do the prediction :
-   In the main.py file, we have written a method for predict in which we are dealing with ModelResolver(). We just need to map the synced models URL in there. Code will predict using that model.

**Interview Que**: How to reinstate the previously deployed image (ie need to dump newly deployed image and restate the old one)

A: Revert the git commit we have done for the latest deployment and do the deployment. This will delete the latest image and reinstate the old one

## Deployment in Heroku cloud

Procfile : specifies some commands that needs to be executed by the app as soon as it starts. It basically indicating, its gives commands to Heroku instance itself that when the entire application is started, what commands needs to run. Commands which are using inside procfile is related to Gunicorn (python http server for wsgi (web server gateway interface) applications which allows us to run python applications concurrently by multiple processes)

In procfile app:app means, ‘app’ comes from app.py and ‘app=Flask(**name**)’ inside app.py while defining flask.

**Refer Heroku documentation for more details on architecture and fundamentals**

There are basically multiple ways for deploying. One approach is using git action, docker and all. Easiest approach is without using CLI ie drag and drop deployment within Heroku. We can create upto 5 different applications for free.

-   **Simple approach**
    -   Login to Heroku Create new app Give name
    -   There are 3 methods for deployment. Heroku Git, Github and container registry
    -   Select Github and give credentials and repo name
    -   If we enable the automatic deploys, whenever there are changes occur in github, it will reflect in Heroku app. Whenever we use CICD pipelines, we used to enable this wrt various pipelines.
    -   We will go for manual deploy. Deploy the main branch
    -   We can check for any error in logs in More option
    -   Run the app.
-   **Configure Github actions**
1.  Create two folders “.github” and within that “workflows”. Create a file “main.yaml”

Reason : As soon as we push the entire files into the repo, github will execute all the workflows as defined inside main.yaml.

1.  Goto github Select repository Settings Secrets Actions New Repository secret
2.  Add HEROKU_API_KEY , HEROKU_APP_NAME and HEROKU_EMAIL. (Heroku API key is in account settings)
3.  Commit and push all files from terminal
4.  Go to the details from the green tick or orange dot to see the actions in details.
5.  Go to Heroku and open app for testing. (Note that once we deployed using github actions and docker, our application in Heroku is running as a container otherwise it would be a python application)

**Flasgger**

Flasgger is a front end UI.

1.  Goto Anaconda prompt pip install flasgger
2.  Make a copy of flask_api.py and import Swagger. (Swagger is just like an API which automatically generates the front end UI part)
3.  Edit the required data as shown in code.
4.  Goto this link ie \<URL\>/apidocs (<http://127.0.0.1:5000/apidocs/>) . Enter the values to test
