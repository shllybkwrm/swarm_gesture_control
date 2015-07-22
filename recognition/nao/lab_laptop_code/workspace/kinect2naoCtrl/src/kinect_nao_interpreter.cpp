/*
 * Paul Bartholomew Feb 2013
 * Preliminary code to read joint angles from skeleton tracking of XBOX Kinect
 *  and translate the information to joint angles that can be read by the NAO
 *  robotic platform
 */

#include "ros/ros.h"
#include "tf/tfMessage.h"
#include "nao_msgs/JointAnglesWithSpeed.h"
#include <string>
#include <vector>
#include <math.h>

//LIST OF VARIABLES
const float PI = 3.14159265359;

//  extraction of kinect joint name
std::string tfJointName;

//  constant "/camera" we don't want to include these elements in the kinect skeletal array
const std::string dontIncludeCameraTFs ("/camera");

// constant "/head" which is the first element published by the kinect
const std::string head ("/head_1");

//  extractions for kinect translational vector data
float X;
float Y;
float Z;

//  arrays which keep all joint names and angles to be sent to NAO robot
std::vector<std::string> naoJointNames(0);
std::vector<float> naoJointAngles(0);

//  array to keep list of kinect joint names
std::vector<std::string> tfNames(0);
//  array to keep list of kinect joint angles
std::vector<float> tfX(0);
std::vector<float> tfY(0);
std::vector<float> tfZ(0);

//  variables for the xyz vectors
//    Left Shoulder
float LS_X, LS_Y, LS_Z;
float LeftShoulderVector_X, LeftShoulderVector_Y, LeftShoulderVector_Z;
float LeftShoulderUnRolled_X, LeftShoulderUnRolled_Y, LeftShoulderUnRolled_Z;
float LeftShoulderUnPitched_X, LeftShoulderUnPitched_Y, LeftShoulderUnPitched_Z;
//    Left Elbow
float LE_X, LE_Y, LE_Z;
float LeftElbowVector_X, LeftElbowVector_Y, LeftElbowVector_Z;
float LeftElbowUnRolled_X, LeftElbowUnRolled_Y, LeftElbowUnRolled_Z;
float LeftElbowUnPitched_X, LeftElbowUnPitched_Y, LeftElbowUnPitched_Z;
float translatedElbow_X, translatedElbow_Y, translatedElbow_Z;
float LeftElbowUnYawed_X, LeftElbowUnYawed_Y, LeftElbowUnYawed_Z;
//    Left Hand
float LH_X, LH_Y, LH_Z;
//    Right Shoulder
float RS_X, RS_Y, RS_Z;
float RightShoulderVector_X, RightShoulderVector_Y, RightShoulderVector_Z;
float RightShoulderUnRolled_X, RightShoulderUnRolled_Y, RightShoulderUnRolled_Z;
float RightShoulderUnPitched_X, RightShoulderUnPitched_Y, RightShoulderUnPitched_Z;
//    Right Elbow
float RE_X, RE_Y, RE_Z;
float RightElbowVector_X, RightElbowVector_Y, RightElbowVector_Z;
float RightElbowUnRolled_X, RightElbowUnRolled_Y, RightElbowUnRolled_Z;
float RightElbowUnPitched_X, RightElbowUnPitched_Y, RightElbowUnPitched_Z;
float translated_R_Elbow_X, translated_R_Elbow_Y, translated_R_Elbow_Z;
float RightElbowUnYawed_X, RightElbowUnYawed_Y, RightElbowUnYawed_Z; 
//    Right Hand
float RH_X, RH_Y, RH_Z;

// variables for NAO angles
float LShoulderRoll, LShoulderPitch, RShoulderRoll, RShoulderPitch;
float LElbowRoll, LElbowYaw, RElbowRoll, RElbowYaw;

// variables for NAO msg parameters
//    fraction of maximum joint velocity [0:1]
float speed;
//    absolute angle (0 is default) or relative change
uint8_t rel;


//Function to convert Kinect Vectors into Pitch and Roll angles for NAO joints
void convertXYZvectorsToAngles()
{
	//get a shoulder, elbow, and hand joints and kinect vectors
	//left shoulder is elements 3
	LS_X = tfX.at(3);
	LS_Y = tfY.at(3);
	LS_Z = tfZ.at(3);
	//left elbow is element 4
	LE_X = tfX.at(4);
	LE_Y = tfY.at(4);
	LE_Z = tfZ.at(4);
	//left hand is element 5
	LH_X = tfX.at(5);
	LH_Y = tfY.at(5);
	LH_Z = tfZ.at(5);
	//right shoulder is element 6
	RS_X = tfX.at(6);
	RS_Y = tfY.at(6);
	RS_Z = tfZ.at(6);
	//right elbow is element 7
	RE_X = tfX.at(7);
	RE_Y = tfY.at(7);
	RE_Z = tfZ.at(7);
	//right hand is element 8
	RH_X = tfX.at(8);
	RH_Y = tfY.at(8);
	RH_Z = tfZ.at(8);

	//show the names of the joints in the tf array
//	for (unsigned i = 3;i<9;i++)
//		ROS_INFO("[%s]", tfNames.at(i).c_str());


////LEFT ARM
	//the kinect vectors point to each of the joints:
	//move the shoulder to the origin
	LeftShoulderVector_X = LE_X-LS_X;
	LeftShoulderVector_Y = LE_Y-LS_Y;
	LeftShoulderVector_Z = LE_Z-LS_Z;
	LeftElbowVector_X = LH_X-LS_X;
	LeftElbowVector_Y = LH_Y-LS_Y;
	LeftElbowVector_Z = LH_Z-LS_Z;	
	
	//change the orientation of the x and z components to match the orientation of the NAO
	LeftShoulderVector_X = -LeftShoulderVector_X;
	LeftShoulderVector_Z = -LeftShoulderVector_Z;
	LeftElbowVector_X = -LeftElbowVector_X;
	LeftElbowVector_Z = -LeftElbowVector_Z;

	//nao defines left arm pitch zero pose as straight forward with pos theta down (backwards of normal rotation)
	LShoulderPitch = atan2(LeftShoulderVector_Z,LeftShoulderVector_X);

	//unPitch the left arm by rotating by Pitch about y-axis
	//NOTE: The value isn't -Pitch since the rotation goes from z-axis with pos theta towards x-axis (backwards on nao)
	//Ry = [ cos(theta)  0  sin(theta)]
	//     [ 0           1          0]
	//     [-sin(theta)  0  cos(theta)]
	LeftShoulderUnPitched_X = cos(LShoulderPitch)*LeftShoulderVector_X+sin(LShoulderPitch)*LeftShoulderVector_Z;
	LeftShoulderUnPitched_Y = LeftShoulderVector_Y;
	LeftShoulderUnPitched_Z = -sin(LShoulderPitch)*LeftShoulderVector_X+cos(LShoulderPitch)*LeftShoulderVector_Z;
	LeftElbowUnPitched_X = cos(LShoulderPitch)*LeftElbowVector_X+sin(LShoulderPitch)*LeftElbowVector_Z;
	LeftElbowUnPitched_Y = LeftElbowVector_Y;
	LeftElbowUnPitched_Z = -sin(LShoulderPitch)*LeftElbowVector_X+cos(LShoulderPitch)*LeftElbowVector_Z;
	
	//nao left shoulder roll sweeps X-Y plane and pitch sweeps X-Z plane
	// Zero roll [-18:76] and Zero pitch [-119.5:119.5] lies on X axis
	//nao defines zero roll as straight forward with pos theta as outward from the body	
	LShoulderRoll  = atan2(LeftShoulderUnPitched_Y,LeftShoulderUnPitched_X);

	//unRoll the left arm by rotating back by -Roll about z-axis
	//Rz = [cos(theta) -sin(theta) 0]
	//     [sin(theta)  cos(theta) 0]
	//     [0           0          1]
	LeftShoulderUnRolled_X = cos(-LShoulderRoll)*LeftShoulderUnPitched_X-sin(-LShoulderRoll)*LeftShoulderUnPitched_Y;
	LeftShoulderUnRolled_Y = sin(-LShoulderRoll)*LeftShoulderUnPitched_X+cos(-LShoulderRoll)*LeftShoulderUnPitched_Y;
	LeftShoulderUnRolled_Z = LeftShoulderUnPitched_Z;
	LeftElbowUnRolled_X = cos(-LShoulderRoll)*LeftElbowUnPitched_X-sin(-LShoulderRoll)*LeftElbowUnPitched_Y;
	LeftElbowUnRolled_Y = sin(-LShoulderRoll)*LeftElbowUnPitched_X+cos(-LShoulderRoll)*LeftElbowUnPitched_Y;
	LeftElbowUnRolled_Z = LeftElbowUnPitched_Z;
	
	//move the unrotated elbow vector to the origin by subtracting the shoulder vector
	translatedElbow_X = LeftElbowUnRolled_X - LeftShoulderUnRolled_X;
	translatedElbow_Y = LeftElbowUnRolled_Y - LeftShoulderUnRolled_Y; //shoulder y is now zero
	translatedElbow_Z = LeftElbowUnRolled_Z - LeftShoulderUnRolled_Z; //shoulder z is now zero

	//change the orientation of the elbow vector to match what the nao is expecting.
	translatedElbow_Y = -translatedElbow_Y;
	
	//get Yaw of elbow (the speck sheet for the elbow yaw is different it shows the z axis pointing up rather than down)
	LElbowYaw = atan2(translatedElbow_Z, translatedElbow_Y);
	
	//un-yaw the elbow by rotating elbow by -yaw about x-axis
	//Rx = [1 0           0         ]
	//     [0 cos(theta) -sin(theta)]
	//     [0 sin(theta)  cos(theta)]
	LeftElbowUnYawed_X = translatedElbow_X;
	LeftElbowUnYawed_Y = cos(-LElbowYaw)*translatedElbow_Y-sin(-LElbowYaw)*translatedElbow_Z;
	LeftElbowUnYawed_Z = sin(-LElbowYaw)*translatedElbow_Y+cos(-LElbowYaw)*translatedElbow_Z;
	
	//get roll angle of elbow
	LElbowRoll = atan2(-LeftElbowUnYawed_Y,LeftElbowUnYawed_X);

	
////RIGHT ARM
	//RightShoulderVector = RE-RS
	RightShoulderVector_X = RE_X-RS_X;
	RightShoulderVector_Y = RE_Y-RS_Y;
	RightShoulderVector_Z = RE_Z-RS_Z;
	RightElbowVector_X = RH_X-RS_X;
	RightElbowVector_Y = RH_Y-RS_Y;
	RightElbowVector_Z = RH_Z-RS_Z;

	//change the orientation of the x and z components to match the orientation of the NAO
	RightShoulderVector_X = -RightShoulderVector_X;
	RightShoulderVector_Z = -RightShoulderVector_Z;
	RightElbowVector_X = -RightElbowVector_X;
	RightElbowVector_Z = -RightElbowVector_Z;
	
	//nao defines right arm pitch zero pose as straight forward with pos theta down
	RShoulderPitch = atan2(RightShoulderVector_Z,RightShoulderVector_X);

	//unPitch the left arm by rotating by Pitch about y-axis
	//NOTE: The value isn't -Pitch since the rotation goes from z-axis with pos theta towards x-axis
	//Ry = [ cos(theta)  0  sin(theta)]
	//	   [ 0           1          0]
	//	   [-sin(theta)  0  cos(theta)]
	RightShoulderUnPitched_X = cos(RShoulderPitch)*RightShoulderVector_X+sin(RShoulderPitch)*RightShoulderVector_Z;
	RightShoulderUnPitched_Y = RightShoulderVector_Y;
	RightShoulderUnPitched_Z = -sin(RShoulderPitch)*RightShoulderVector_X+cos(RShoulderPitch)*RightShoulderVector_Z;
	RightElbowUnPitched_X = cos(RShoulderPitch)*RightElbowVector_X+sin(RShoulderPitch)*RightElbowVector_Z;
	RightElbowUnPitched_Y = RightElbowVector_Y;
	RightElbowUnPitched_Z = -sin(RShoulderPitch)*RightElbowVector_X+cos(RShoulderPitch)*RightElbowVector_Z;

	//nao right shoulder roll sweeps Y-Z plane and pitch sweeps X-Z plane
	// Zero roll [-76:18] and Zero pitch [-119.5:119.5] lies on Z axis
	//nao defines zero roll as straight forward with pos theta as inward to the body	
	RShoulderRoll  = atan2(RightShoulderUnPitched_Y,RightShoulderUnPitched_X);

	//unRoll the right arm by rotating back by -Roll about z-axis
	//Rz = [cos(theta) -sin(theta) 0]
	//	   [sin(theta)  cos(theta) 0]
	//	   [0           0          1]
	RightShoulderUnRolled_X = cos(-RShoulderRoll)*RightShoulderUnPitched_X-sin(-RShoulderRoll)*RightShoulderUnPitched_Y;
	RightShoulderUnRolled_Y = sin(-RShoulderRoll)*RightShoulderUnPitched_X+cos(-RShoulderRoll)*RightShoulderUnPitched_Y;
	RightShoulderUnRolled_Z = RightShoulderUnPitched_Z;
	RightElbowUnRolled_X = cos(-RShoulderRoll)*RightElbowUnPitched_X-sin(-RShoulderRoll)*RightElbowUnPitched_Y;
	RightElbowUnRolled_Y = sin(-RShoulderRoll)*RightElbowUnPitched_X+cos(-RShoulderRoll)*RightElbowUnPitched_Y;
	RightElbowUnRolled_Z = RightElbowUnPitched_Z;
	
	//move the unrotated elbow vector to the origin by subtracting the shoulder vector
	translated_R_Elbow_X = RightElbowUnRolled_X - RightShoulderUnRolled_X;
	translated_R_Elbow_Y = RightElbowUnRolled_Y - RightShoulderUnRolled_Y; //shoulder y is now zero
	translated_R_Elbow_Z = RightElbowUnRolled_Z - RightShoulderUnRolled_Z; //shoulder z is now zero
	
	//change the orientation of the elbow vector to match what the nao is expecting.
	translated_R_Elbow_Z = -translated_R_Elbow_Z;

	//get Yaw of right elbow (the speck sheet for the elbow yaw is different it shows the y axis pointing in rather than out)
	RElbowYaw = atan2(translated_R_Elbow_Z, translated_R_Elbow_Y);	

	//un-yaw the elbow by rotating elbow by -yaw about x-axis
	//Rx = [1 0           0         ]
	//     [0 cos(theta) -sin(theta)]
	//     [0 sin(theta)  cos(theta)]
	RightElbowUnYawed_X = translated_R_Elbow_X;
	RightElbowUnYawed_Y = cos(-RElbowYaw)*translated_R_Elbow_Y - sin(-RElbowYaw)*translated_R_Elbow_Z;
	RightElbowUnYawed_Z = sin(-RElbowYaw)*translated_R_Elbow_Y + cos(-RElbowYaw)*translated_R_Elbow_Z;

	//get roll angle of right elbow
	RElbowRoll = atan2(RightElbowUnYawed_Y, RightElbowUnYawed_X);

	//begin by emptying the name and angle arrays
	while(!naoJointNames.empty())
	{
		naoJointNames.pop_back(); 
		naoJointAngles.pop_back();		
	}
	
	//show the joint angles (in degrees) to be published
//	ROS_INFO("LS_Roll %f :LS_Pitch %f :RS_Roll %f :RS_Pitch %f", LShoulderRoll*180/PI, LShoulderPitch*180/PI, RShoulderRoll*180/PI, RShoulderPitch*180/PI);
//	ROS_INFO("LE_Yaw %f :LE_Roll %f :RE_Yaw %f :RE_Roll %f", LElbowYaw*180/PI, LElbowRoll*180/PI, RElbowYaw*180/PI, RElbowRoll*180/PI);
	
	//build up name and angle arrays with correspoinding fields
	naoJointNames.push_back("LShoulderRoll");
	naoJointAngles.push_back(LShoulderRoll);
	naoJointNames.push_back("LShoulderPitch");
	naoJointAngles.push_back(LShoulderPitch);
	naoJointNames.push_back("LElbowYaw");
	naoJointAngles.push_back(LElbowYaw);	
	naoJointNames.push_back("LElbowRoll");
	naoJointAngles.push_back(LElbowRoll);
	naoJointNames.push_back("RShoulderRoll");
	naoJointAngles.push_back(RShoulderRoll);
	naoJointNames.push_back("RShoulderPitch");
	naoJointAngles.push_back(RShoulderPitch);
	naoJointNames.push_back("RElbowYaw");
	naoJointAngles.push_back(RElbowYaw);	
	naoJointNames.push_back("RElbowRoll");
	naoJointAngles.push_back(RElbowRoll);
	
}


//Subscriber: function called each time the Kinect publishes new skeletal data
void getTFvectors(const tf::tfMessage::ConstPtr& msg)
{
	//get the joint name
	tfJointName = msg->transforms[0].child_frame_id;
	//check to see if the joint is a "/camera..." element
	if(tfJointName.compare(0,7,dontIncludeCameraTFs)!=0)
	{
		//get the X, Y, and Z coordinates for the kinect joint
		X = msg->transforms[0].transform.translation.x;
		Y = msg->transforms[0].transform.translation.y;
		Z = msg->transforms[0].transform.translation.z;
		
		//The head element is the first element the kinect will publish
		//  it will allow us to begin our parsing
		if(tfJointName.compare(head)==0)
		{
			//the head element is the start of a new kinect body pose
			//so call a function to empty the vector and transform the vectors
			//into angles theta and phi
			//wait for list to be length of 15 before making function call
			if(tfNames.size()==15)
			{
				//New head element means we should publish the old kinect
				//  skeleton and clear out the array before starting again
				convertXYZvectorsToAngles();
			}

			//empty the X-Y-Z coordinate arrays	
			while(!tfNames.empty())
			{
				tfNames.pop_back();
				tfX.pop_back();
				tfY.pop_back();
				tfZ.pop_back();
			}
			//store "/head" element and xyz vector
			tfNames.push_back(tfJointName);
			tfX.push_back(X);
			tfY.push_back(Y);
			tfZ.push_back(Z);
			
		}
		//if the element isn't "/head" or "/camera.." store it in order it appears
		else
		{
			tfNames.push_back(tfJointName);
			tfX.push_back(X);
			tfY.push_back(Y);
			tfZ.push_back(Z);
		}
		
		//troubleshoot which elements are being stored
//		ROS_INFO("[%s]", tfJointName.c_str());
//		ROS_INFO("[%f]", X);
//		ROS_INFO("[%f]", Y);
//		ROS_INFO("[%f]", Z);

	}

}

void initializeArms()
{
	// initialize all the joints to be controlled to zero
	naoJointNames.push_back("LShoulderRoll");
	naoJointAngles.push_back(0.0);
	naoJointNames.push_back("LShoulderPitch");
	naoJointAngles.push_back(0.0);
	naoJointNames.push_back("LElbowYaw");
	naoJointAngles.push_back(0.0);
	naoJointNames.push_back("LElbowRoll");
	naoJointAngles.push_back(0.0);
	naoJointNames.push_back("RShoulderRoll");
	naoJointAngles.push_back(0.0);
	naoJointNames.push_back("RShoulderPitch");
	naoJointAngles.push_back(0.0);
	naoJointNames.push_back("RElbowYaw");
	naoJointAngles.push_back(0.0);	
	naoJointNames.push_back("RElbowRoll");
	naoJointAngles.push_back(0.0);
	
}

int main(int argc, char **argv)
{

	//initialize a node with name
	ros::init(argc, argv, "Kinect2Nao");
	
	//create node handle
	ros::NodeHandle n;
	
	//create a function to subscribe to a topic
	ros::Subscriber sub = n.subscribe("tf", 1000, getTFvectors);	
	
	//create a function to advertise on a given topic
	ros::Publisher joint_angles_pub = n.advertise<nao_msgs::JointAnglesWithSpeed>("joint_angles",1000);

	//choose the looping rate
	ros::Rate loop_rate(30.0);
	
	//create message element to be filled with appropriate data to be published
	nao_msgs::JointAnglesWithSpeed msg;

	//initialize arms at zero;
	initializeArms();

	//loop
	while(ros::ok())
	{	
		
		//Put elements into message for publishing topic
		msg.joint_names = naoJointNames; //string[] -From Nao Datasheet (must be array)
		msg.joint_angles = naoJointAngles; //float[] -In Radians (must be array)
		speed = 0.5;
		rel = 0;				
		msg.speed = speed; //float
		msg.relative = rel; //uint8 

		//publish
    		joint_angles_pub.publish(msg);
		
		//for troubleshooting print the converted names and angles	
//		for(unsigned i=0;i<naoJointNames.size();i++)		
//			ROS_INFO("[%s]: [%f]",naoJointNames.at(i).c_str(),naoJointAngles.at(i));

		//spin once
		ros::spinOnce();

		//sleep
		loop_rate.sleep();
	}
	return 0;
}
