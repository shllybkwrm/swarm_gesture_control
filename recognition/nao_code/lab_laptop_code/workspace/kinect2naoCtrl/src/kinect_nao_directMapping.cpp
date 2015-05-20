/*
 * Paul Bartholomew Feb 2014
 *  Read each joint angle as published by the findJointAngles routine, and give them directly to the NAO
 */

#include "ros/ros.h"
#include "nao_msgs/JointAnglesWithSpeed.h"
#include <string>
#include <vector>
#include <math.h>

////LIST OF VARIABLES
//  arrays which keep all joint names and angles to be sent to NAO robot
std::vector<std::string> naoJointNames(0);
std::vector<float> naoJointAngles(0);
// variables for NAO msg parameters
//    fraction of maximum joint velocity [0:1]
float speed;
//    absolute angle (0 is default) or relative change
uint8_t rel;


//Direct Mapping of Joint Angles
void DirectMap(const nao_msgs::JointAnglesWithSpeed::ConstPtr& msg)
{
	naoJointNames = msg->joint_names;
	naoJointAngles = msg->joint_angles;	
}


//Main Method
int main(int argc, char **argv)
{
	//initialize a node with name
	ros::init(argc, argv, "DirectMap");
	
	//create node handle
	ros::NodeHandle n;
	
	//create a function to subscribe to a topic
	ros::Subscriber sub = n.subscribe("raw_joint_angles", 1000, DirectMap);	
	
	//create a function to advertise on a given topic
	ros::Publisher joint_angles_pub = n.advertise<nao_msgs::JointAnglesWithSpeed>("joint_angles",1000);
	
	//choose the looping rate
	ros::Rate loop_rate(30.0);
	
	//create message element to be filled with appropriate data to be published
	nao_msgs::JointAnglesWithSpeed msg;

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

		//spin once
		ros::spinOnce();

		//sleep
		loop_rate.sleep();
	}
	return 0;
}
