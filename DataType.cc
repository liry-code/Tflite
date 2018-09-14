#include "DataType.h"
#include <iostream>
#include <cstring>
using namespace std;

int DT::getDataTypeBit(){
//
    const char* typeName = dataTypeName.c_str();
	//this->DataTypeName = str;
	if(strcmp("i",typeName)==0){
      return 3;
    }
    if (strcmp("l",typeName)==0){
      return 8;
    }
    if (strcmp("f",typeName)==0){
      return 4;
    }
    if(strcmp("c",typeName)==0){
      return 1; 
    }
}

void DT::setDataTypeName(string str){
	dataTypeName = str;
}
