/*
 * Paul Bartholomew Feb 2014
 *  Use the method shown in Optimal Behavior Composition Thesis to mimic human pose on the NAO
 *  robotic platform
 */

#include "ros/ros.h"
#include "nao_msgs/JointAnglesWithSpeed.h"
#include "geometry_msgs/Twist.h"
#include <string>
#include <vector>
#include <math.h>
#include <Eigen/Dense>

//Number of predifinded motion primitives
#define numOfPrimitives 5

using namespace Eigen;

////LIST OF VARIABLES
//Input Joint Angles from Kinect
std::vector<float> InputJointAngles(0);
//Output Joint Angles from Optimal Behavior Composition
std::vector<float> OutputJointAngles(0);
//Left Shoulder Pitch Angles
std::vector<float> LSP_display(0);
//Input Joint Names from Kinect
std::vector<std::string> InputJointNames(0);
//Output Joint Names from Optimal Behavior Composition
std::vector<std::string> OutputJointNames(0);
//How many joints positions were sent from Kinect
int numOfJoints;
//NAO msg parameters for fraction of maximum joint velocity [0:1]
float speed;
//NAO msg parameters for absolute angle (0 is default) or relative change
uint8_t rel;
//Time keeping variables
double beginningTime, elapsedTime, mimicryTime;
bool ready = false;
//Keep Time
std::vector<double> clockTime(0);
//set interval for mimicry fitting of Optimal Behavior Composition Method
ros::Duration fixedMimicryInterval(5.0);
//Optimal behavior weights
std::vector<float> weights(0);
//Inner product of captured human motion with robot motion primitives
MatrixXf PHI(numOfPrimitives,1);
//Inner product of motion primitives with all other motion primitives
MatrixXf PSI(numOfPrimitives,numOfPrimitives);

//Define Motion Primitives
MatrixXf primitives(double time)
{
	MatrixXf motionPrimitive(numOfPrimitives,1);
	motionPrimitive(0) = cos(0.25*time);
	motionPrimitive(1) = cos(0.5*time);
	motionPrimitive(2) = cos(1.0*time);
	motionPrimitive(3) = cos(2.0*time);
	motionPrimitive(4) = cos(4.0*time);
	return motionPrimitive;
}

//Function to initialize (zero) matricies
void initializeArrays()
{
	for (int index1=0; index1<numOfPrimitives; index1++)
	{
		PHI(index1)= 0.0;
		for (int index2=0; index2<numOfPrimitives; index2++)
		{
			PSI(index1,index2) = 0.0;
		}
	}
}

//Thesis Derivation Code
void OptBehaveComp(const nao_msgs::JointAnglesWithSpeed::ConstPtr& msg)
{
	elapsedTime = ros::Time::now().toSec()-beginningTime;
	//ONCE THE APPROPRIATE TIME HAS ELAPSED
	if(elapsedTime>mimicryTime)
	{
		//increment next time for mimicry
		mimicryTime += fixedMimicryInterval.toSec(); 
		
		// Caluculate the optimal weights with inverse PSI and PHI
		//  [        ]   [           ]^-1   [     ]
		//  [ weight ] = [    PSI    ]    * [ PHI ]
		//  [        ]   [           ]      [     ]
		//     nx1            nxn             nx1
		MatrixXf OptWeights = (PSI.inverse())*PHI;
		//save the current time of the weight computation
		weights.insert(weights.begin(),(float)elapsedTime);
		//save the weight values
		for (int i=numOfPrimitives-1; i>=0; i--)
		{
			weights.insert(weights.begin(),OptWeights(i));
		}
		initializeArrays();
		ready = true;
	}
	//Save the current time to be used for the exectution after Optimial Behavior Composition has met its timing constraints
	clockTime.insert(clockTime.begin(),elapsedTime);

	//Update the primitive values for the current time
	MatrixXf primitiveValues = primitives(elapsedTime);

	//Get the current angle positions and corresponding names
	InputJointAngles = msg->joint_angles;
	numOfJoints = InputJointAngles.size();
	InputJointNames  = msg->joint_names;

	LSP_display.insert(LSP_display.begin(),InputJointAngles.at(1));
	////Use Optimal Behavoir Composition to find the best weightings for the motion primitives
	
	//Find PHI
	for (int index1=0; index1<numOfPrimitives; index1++)
	{
		PHI(index1) = PHI(index1) + primitiveValues(index1)*InputJointAngles.at(1); //LSPitch
	}
	//Find PSI
	for (int index1=0; index1<numOfPrimitives; index1++)
	{
		for (int index2=0; index2<numOfPrimitives; index2++)
		{
			PSI(index1,index2) = PSI(index1,index2) + primitiveValues(index1)*primitiveValues(index2);
		}
	}	
}


void updateOutputAngles()
{
	int size = OutputJointAngles.size();
	for (int index=0; index<size; index++)
	{
		OutputJointAngles.pop_back();
		OutputJointNames.pop_back();
	}
	//what time to execute commands (retrieved elapsed Time)
	double timeToExecute = clockTime.back();
	clockTime.pop_back();	
	//Determine which set of weights to use by the time the set of weights were computed
	if(((double)weights.back())<=timeToExecute)
	{
		for(int i=0;i<numOfPrimitives+1;i++)
		{
			weights.pop_back();
		}
	}
	if(!(weights.empty()))
	{
		//
		int endLocation = weights.size()-1;
		//Initialize the weights matrix
		MatrixXf OptWeights(1,5);
		for (int i=numOfPrimitives-1;i>=0;i--)
		{
			//decrement pointer
			endLocation--;
			//build a matrix in reverse order (4 down to zero)
			OptWeights(0,i)=weights.at(endLocation);
		}

		//get the sinusoidal values for the past mimicry time
		MatrixXf composition = OptWeights*primitives(timeToExecute);

		float output = composition(0,0);
		//Add a PID controller to avoid discontinuities

		//Output composed motion
		OutputJointAngles.push_back(output);	
		OutputJointNames.push_back("LShoulderPitch");
	}
	else
	{
		ready=false;
	}
}

//Main Method
int main(int argc, char **argv)
{
	//initialize a node with name
	ros::init(argc, argv, "OptBehavComp");
	//create node handle (must be first command)
	ros::NodeHandle n;

	//initialize the durration (time) counter
	beginningTime = ros::Time::now().toSec();
	mimicryTime = fixedMimicryInterval.toSec();

	//initialize arrays PHI and PSI to zero
	initializeArrays();
	
	//create a function to subscribe to a topic
	ros::Subscriber sub = n.subscribe("raw_joint_angles", 1000, OptBehaveComp);	
	
	//create a function to advertise on a given topic
	ros::Publisher joint_angles_pub = n.advertise<nao_msgs::JointAnglesWithSpeed>("joint_angles",1000);
	ros::Publisher pub2 = n.advertise<geometry_msgs::Twist>("OBC",1000);

	//choose the looping rate
	ros::Rate loop_rate(30.0);
	
	//create message element to be filled with appropriate data to be published
	nao_msgs::JointAnglesWithSpeed msg;
	geometry_msgs::Twist msg2;

	//loop
	while(ros::ok())
	{
		//Update the optimized output angles
		if(ready)
		{
			updateOutputAngles();
		}
		
		//Put elements into message for publishing topic
		msg.joint_names  = OutputJointNames; //string[] -From Nao Datasheet (must be array)
		msg.joint_angles = OutputJointAngles; //float[] -In Radians (must be array)
		speed = 0.5;
		rel = 0;				
		msg.speed = speed; //float
		msg.relative = rel; //uint8 

		if(ready)
		{
			msg2.angular.x = LSP_display.back();
			LSP_display.pop_back();
			msg2.angular.y = OutputJointAngles.at(0);

			//only publish data when ready
			pub2.publish(msg2);
		}

		//publish
    		joint_angles_pub.publish(msg);
		

		//spin once
		ros::spinOnce();

		//sleep
		loop_rate.sleep();
	}
	return 0;
}
