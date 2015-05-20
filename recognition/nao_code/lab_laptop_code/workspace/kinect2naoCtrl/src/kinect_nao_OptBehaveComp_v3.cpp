/*
 * Paul Bartholomew Feb 2014
 *  Use the method shown in Optimal Behavior Composition Thesis to mimic human pose on the NAO
 *  robotic platform
 *
 * Version 3 adds P-I controller
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
//Input Joint Names from Kinect
std::vector<std::string> InputJointNames(0);
//Output Joint Names from Optimal Behavior Composition
std::vector<std::string> OutputJointNames(0);
//How many joints positions were sent from Kinect
int numOfJoints;
//NAO msg parameters for fraction of maximum joint velocity [0:1]
float speed = 0.5;
//NAO msg parameters for absolute angle (0 is default) or relative change
uint8_t rel = 0;
//Time keeping variables
double beginningTime, elapsedTime, mimicryTime;
bool ready = false;
//Keep Time
std::vector<double> clockTime(0);
//set interval for mimicry fitting of Optimal Behavior Composition Method
ros::Duration fixedMimicryInterval(5.0);
//Optimal behavior weights
std::vector<float> weights_LSP(0); //Left Shoulder Pitch
std::vector<float> weights_LSR(0); //Left Shoulder Roll
std::vector<float> weights_LEY(0); //Left Elbow Yaw
std::vector<float> weights_LER(0); //Left Elbow Roll
std::vector<float> weights_RSP(0); //Right Shoulder Pitch
std::vector<float> weights_RSR(0); //Right Shoulder Roll
std::vector<float> weights_REY(0); //Right Elbow Yaw
std::vector<float> weights_RER(0); //Right Elbow Roll
//Inner product of captured human motion with robot motion primitives
MatrixXf PHI_LSP(numOfPrimitives,1); //Left Shoulder Pitch
MatrixXf PHI_LSR(numOfPrimitives,1); //Left Shoulder Roll
MatrixXf PHI_LEY(numOfPrimitives,1); //Left Elbow Yaw
MatrixXf PHI_LER(numOfPrimitives,1); //Left Elbow Roll
MatrixXf PHI_RSP(numOfPrimitives,1); //Right Shoulder Pitch
MatrixXf PHI_RSR(numOfPrimitives,1); //Right Shoulder Roll
MatrixXf PHI_REY(numOfPrimitives,1); //Right Elbow Yaw
MatrixXf PHI_RER(numOfPrimitives,1); //Right Elbow Roll
//Inner product of motion primitives with all other motion primitives
MatrixXf PSI(numOfPrimitives,numOfPrimitives);
//Left Shoulder Pitch Angles
std::vector<float> LSP_display(0);
//Left Shoulder Roll Angles
std::vector<float> LSR_display(0);
//Left Elbow Yaw Angles
std::vector<float> LEY_display(0);
//Left Elbow Roll Angles
std::vector<float> LER_display(0);
//Right Shoulder Pitch Angles
std::vector<float> RSP_display(0);
//Right Shoulder Roll Angles
std::vector<float> RSR_display(0);
//Right Elbow Yaw Angles
std::vector<float> REY_display(0);
//Right Elbow Roll Angles
std::vector<float> RER_display(0);
//PI controller values
const float KP=1.0;//0.15;
const float KI=0.0;//0.00001;
//PI contorller Left arm process values
float processValue_LSP,processValue_LSR,processValue_LEY,processValue_LER;
//PI controller Right arm process values
float processValue_RSP,processValue_RSR,processValue_REY,processValue_RER;
//PI controlller Error value
float error_LSP = 0.0;
float error_LSR = 0.0;
float error_LEY = 0.0;
float error_LER = 0.0;
float error_RSP = 0.0;
float error_RSR = 0.0;
float error_REY = 0.0;
float error_RER = 0.0;
//PI controlller Last Error values NOTE: only needed for derivative
//float lastError_LSP, lastError_LSR, lastError_LEY, lastError_LER, lastError_RSP, lastError_RSR, lastError_REY, lastError_RER;
//PI controller Process Values
float regulated_LSP = 0.0;
float regulated_LSR = 0.0;
float regulated_LEY = 0.0;
float regulated_LER = 0.0;
float regulated_RSP = 0.0;
float regulated_RSR = 0.0;
float regulated_REY = 0.0;
float regulated_RER = 0.0;
//PI controller Integral Error values
float integral_LSP = 0.0;
float integral_LSR = 0.0;
float integral_LEY = 0.0;
float integral_LER = 0.0;
float integral_RSP = 0.0;
float integral_RSR = 0.0;
float integral_REY = 0.0;
float integral_RER = 0.0;

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
		PHI_LSP(index1)= 0.0;
		PHI_LSR(index1)= 0.0;
		PHI_LEY(index1)= 0.0;
		PHI_LER(index1)= 0.0;
		PHI_RSP(index1)= 0.0;
		PHI_RSR(index1)= 0.0;
		PHI_REY(index1)= 0.0;
		PHI_RER(index1)= 0.0;
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
		MatrixXf PSI_inv = PSI.inverse();
		MatrixXf OptWeights_LSP = PSI_inv*PHI_LSP;
		MatrixXf OptWeights_LSR = PSI_inv*PHI_LSR;
		MatrixXf OptWeights_LEY = PSI_inv*PHI_LEY;
		MatrixXf OptWeights_LER = PSI_inv*PHI_LER;
		MatrixXf OptWeights_RSP = PSI_inv*PHI_RSP;
		MatrixXf OptWeights_RSR = PSI_inv*PHI_RSR;
		MatrixXf OptWeights_REY = PSI_inv*PHI_REY;
		MatrixXf OptWeights_RER = PSI_inv*PHI_RER;

		//save the current time of the weight computation
		weights_LSP.insert(weights_LSP.begin(),(float)elapsedTime);
		weights_LSR.insert(weights_LSR.begin(),(float)elapsedTime);
		weights_LEY.insert(weights_LEY.begin(),(float)elapsedTime);
		weights_LER.insert(weights_LER.begin(),(float)elapsedTime);
		weights_RSP.insert(weights_RSP.begin(),(float)elapsedTime);
		weights_RSR.insert(weights_RSR.begin(),(float)elapsedTime);
		weights_REY.insert(weights_REY.begin(),(float)elapsedTime);
		weights_RER.insert(weights_RER.begin(),(float)elapsedTime);

		//save the weight values
		for (int i=numOfPrimitives-1; i>=0; i--)
		{
			weights_LSP.insert(weights_LSP.begin(), OptWeights_LSP(i));
			weights_LSR.insert(weights_LSR.begin(), OptWeights_LSR(i));
			weights_LEY.insert(weights_LEY.begin(), OptWeights_LEY(i));
			weights_LER.insert(weights_LER.begin(), OptWeights_LER(i));
			weights_RSP.insert(weights_RSP.begin(), OptWeights_RSP(i));
			weights_RSR.insert(weights_RSR.begin(), OptWeights_RSR(i));
			weights_REY.insert(weights_REY.begin(), OptWeights_REY(i));
			weights_RER.insert(weights_RER.begin(), OptWeights_RER(i));
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

	LSP_display.insert(LSP_display.begin(),InputJointAngles.at(0));
	LSR_display.insert(LSR_display.begin(),InputJointAngles.at(1));
	LEY_display.insert(LEY_display.begin(),InputJointAngles.at(2));
	LER_display.insert(LER_display.begin(),InputJointAngles.at(3));
	RSP_display.insert(RSP_display.begin(),InputJointAngles.at(4));
	RSR_display.insert(RSR_display.begin(),InputJointAngles.at(5));
	REY_display.insert(REY_display.begin(),InputJointAngles.at(6));
	RER_display.insert(RER_display.begin(),InputJointAngles.at(7));

	////Use Optimal Behavoir Composition to find the best weightings for the motion primitives
	
	//Find PHI
	for (int index1=0; index1<numOfPrimitives; index1++)
	{
		PHI_LSP(index1) = PHI_LSP(index1) + primitiveValues(index1)*InputJointAngles.at(0); //LSPitch
		PHI_LSR(index1) = PHI_LSR(index1) + primitiveValues(index1)*InputJointAngles.at(1); //LSRoll
		PHI_LEY(index1) = PHI_LEY(index1) + primitiveValues(index1)*InputJointAngles.at(2); //LEYaw
		PHI_LER(index1) = PHI_LER(index1) + primitiveValues(index1)*InputJointAngles.at(3); //LERoll
		PHI_RSP(index1) = PHI_RSP(index1) + primitiveValues(index1)*InputJointAngles.at(4); //RSPitch
		PHI_RSR(index1) = PHI_RSR(index1) + primitiveValues(index1)*InputJointAngles.at(5); //RSRoll
		PHI_REY(index1) = PHI_REY(index1) + primitiveValues(index1)*InputJointAngles.at(6); //REYaw
		PHI_RER(index1) = PHI_RER(index1) + primitiveValues(index1)*InputJointAngles.at(7); //RERoll		
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

	//Determine which set of weights to use by the time the set of weights were computed
	if(((double)weights_LSP.back())<=timeToExecute)
	{
		for(int i=0;i<numOfPrimitives+1;i++)
		{
			weights_LSP.pop_back();
			weights_LSR.pop_back();
			weights_LEY.pop_back();
			weights_LER.pop_back();
			weights_RSP.pop_back();
			weights_RSR.pop_back();
			weights_REY.pop_back();
			weights_RER.pop_back();
		}
	}
	if(!(weights_LSP.empty()))
	{
		//Upon execution pop the execution time off the stack
		clockTime.pop_back();
		//
		int endLocation = weights_LSP.size()-1;
		//Initialize the weights matricies
		MatrixXf OptWeights_LSP(1,5);
		MatrixXf OptWeights_LSR(1,5);
		MatrixXf OptWeights_LEY(1,5);
		MatrixXf OptWeights_LER(1,5);
		MatrixXf OptWeights_RSP(1,5);
		MatrixXf OptWeights_RSR(1,5);
		MatrixXf OptWeights_REY(1,5);
		MatrixXf OptWeights_RER(1,5);
	

		for (int i=numOfPrimitives-1;i>=0;i--)
		{
			//decrement pointer
			endLocation--;
			//build a matrix in reverse order (4 down to zero)
			OptWeights_LSP(0,i)=weights_LSP.at(endLocation);
			OptWeights_LSR(0,i)=weights_LSR.at(endLocation);
			OptWeights_LEY(0,i)=weights_LEY.at(endLocation);
			OptWeights_LER(0,i)=weights_LER.at(endLocation);
			OptWeights_RSP(0,i)=weights_RSP.at(endLocation);
			OptWeights_RSR(0,i)=weights_RSR.at(endLocation);
			OptWeights_REY(0,i)=weights_REY.at(endLocation);
			OptWeights_RER(0,i)=weights_RER.at(endLocation);
		}

		//get the sinusoidal values for the past mimicry time
		MatrixXf composition_LSP = OptWeights_LSP*primitives(timeToExecute);
		MatrixXf composition_LSR = OptWeights_LSR*primitives(timeToExecute);
		MatrixXf composition_LEY = OptWeights_LEY*primitives(timeToExecute);
		MatrixXf composition_LER = OptWeights_LER*primitives(timeToExecute);
		MatrixXf composition_RSP = OptWeights_RSP*primitives(timeToExecute);
		MatrixXf composition_RSR = OptWeights_RSR*primitives(timeToExecute);
		MatrixXf composition_REY = OptWeights_REY*primitives(timeToExecute);
		MatrixXf composition_RER = OptWeights_RER*primitives(timeToExecute);

		float desired_LSP = composition_LSP(0,0);
		float desired_LSR = composition_LSR(0,0);
		float desired_LEY = composition_LEY(0,0);
		float desired_LER = composition_LER(0,0);
		float desired_RSP = composition_RSP(0,0);
		float desired_RSR = composition_RSR(0,0);
		float desired_REY = composition_REY(0,0);
		float desired_RER = composition_RER(0,0);

		////Add a PID controller to avoid discontinuities
		//Left Shoulder Pitch
		//lastError_LSP = error_LSP;
		error_LSP = desired_LSP-regulated_LSP;
		integral_LSP = integral_LSP + error_LSP;
		regulated_LSP = KP*error_LSP + KI*integral_LSP;
		//Left Shoulder Roll
		//lastError_LSR = error_LSR;
		error_LSR = desired_LSR-regulated_LSR;
		integral_LSR = integral_LSR + error_LSR;
		regulated_LSR = KP*error_LSR + KI*integral_LSR;
		//Left Elbow Yaw
		//lastError_LEY = error_LEY;
		error_LEY = desired_LEY-regulated_LEY;
		integral_LEY = integral_LEY + error_LEY;
		regulated_LEY = KP*error_LEY + KI*integral_LEY;
		//Left Elbow Roll
		//lastError_LER = error_LER;
		error_LER = desired_LER-regulated_LER;
		integral_LER = integral_LER + error_LER;
		regulated_LER = KP*error_LER + KI*integral_LER;
		//Right Shoulder Pitch
		//lastError_RSP = error_RSP;
		error_RSP = desired_RSP-regulated_RSP;
		integral_RSP = integral_RSP + error_RSP;
		regulated_RSP = KP*error_RSP + KI*integral_RSP;
		//Right Shoulder Roll
		//lastError_RSR = error_RSR;
		error_RSR = desired_RSR-regulated_RSR;
		integral_RSR = integral_RSR + error_RSR;
		regulated_RSR = KP*error_RSR + KI*integral_RSR;
		//Right Elbow Yaw
		//lastError_REY = error_REY;
		error_REY = desired_REY-regulated_REY;
		integral_REY = integral_REY + error_REY;
		regulated_REY = KP*error_REY + KI*integral_REY;
		//Right Elbow Roll
		//lastError_RER = error_RER;
		error_RER = desired_RER-regulated_RER;
		integral_RER = integral_RER + error_RER;
		regulated_RER = KP*error_RER + KI*integral_RER;

		//Output composed motion
		OutputJointNames.push_back("LShoulderPitch");
		OutputJointAngles.push_back(regulated_LSP);	
		OutputJointNames.push_back("LShoulderRoll");
		OutputJointAngles.push_back(regulated_LSR);
		OutputJointNames.push_back("LElbowYaw");
		OutputJointAngles.push_back(regulated_LEY);
		OutputJointNames.push_back("LElbowRoll");
		OutputJointAngles.push_back(regulated_LER);
		OutputJointNames.push_back("RShoulderPitch");
		OutputJointAngles.push_back(regulated_RSP);	
		OutputJointNames.push_back("RShoulderRoll");
		OutputJointAngles.push_back(regulated_RSR);
		OutputJointNames.push_back("RElbowYaw");
		OutputJointAngles.push_back(regulated_REY);
		OutputJointNames.push_back("RElbowRoll");
		OutputJointAngles.push_back(regulated_RER);

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
	ros::Publisher pub_LSP = n.advertise<geometry_msgs::Twist>("OBC_LSP",100);
	ros::Publisher pub_LSR = n.advertise<geometry_msgs::Twist>("OBC_LSR",100);
	ros::Publisher pub_LEY = n.advertise<geometry_msgs::Twist>("OBC_LEY",100);
	ros::Publisher pub_LER = n.advertise<geometry_msgs::Twist>("OBC_LER",100);
	ros::Publisher pub_RSP = n.advertise<geometry_msgs::Twist>("OBC_RSP",100);
	ros::Publisher pub_RSR = n.advertise<geometry_msgs::Twist>("OBC_RSR",100);
	ros::Publisher pub_REY = n.advertise<geometry_msgs::Twist>("OBC_REY",100);
	ros::Publisher pub_RER = n.advertise<geometry_msgs::Twist>("OBC_RER",100);

	//choose the looping rate
	ros::Rate loop_rate(30.0);
	
	//create message element to be filled with appropriate data to be published
	nao_msgs::JointAnglesWithSpeed msg;
	//create message for visualizing input reference and output Optimimal Behavior Composition
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
		msg.joint_names  = OutputJointNames;  //string[] -From Nao Datasheet (must be array)
		msg.joint_angles = OutputJointAngles; //float[] -In Radians (must be array)
		msg.speed = speed;                    //float
		msg.relative = rel;                   //uint8 

		if(ready)
		{
			msg2.angular.x = LSP_display.back();
			LSP_display.pop_back();
			msg2.angular.y = OutputJointAngles.at(0);
			//only publish data when ready
			pub_LSP.publish(msg2);

			msg2.angular.x = LSR_display.back();
			LSR_display.pop_back();
			msg2.angular.y = OutputJointAngles.at(1);
			//only publish data when ready
			pub_LSR.publish(msg2);

			msg2.angular.x = LEY_display.back();
			LEY_display.pop_back();
			msg2.angular.y = OutputJointAngles.at(2);
			//only publish data when ready
			pub_LEY.publish(msg2);

			msg2.angular.x = LER_display.back();
			LER_display.pop_back();
			msg2.angular.y = OutputJointAngles.at(3);
			//only publish data when ready
			pub_LER.publish(msg2);

			msg2.angular.x = RSP_display.back();
			RSP_display.pop_back();
			msg2.angular.y = OutputJointAngles.at(4);
			//only publish data when ready
			pub_RSP.publish(msg2);

			msg2.angular.x = RSR_display.back();
			RSR_display.pop_back();
			msg2.angular.y = OutputJointAngles.at(5);
			//only publish data when ready
			pub_RSR.publish(msg2);

			msg2.angular.x = REY_display.back();
			REY_display.pop_back();
			msg2.angular.y = OutputJointAngles.at(6);
			//only publish data when ready
			pub_REY.publish(msg2);

			msg2.angular.x = RER_display.back();
			RER_display.pop_back();
			msg2.angular.y = OutputJointAngles.at(7);
			//only publish data when ready
			pub_RER.publish(msg2);
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
