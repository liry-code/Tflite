#include <iostream>
#include <string>
#include <exception>
using namespace std;

/** Type of elements in a {@link TfLiteTensor}. */
enum DataType {
  /** float. */
  f=1,

  /** int. */
  i=2,

  /** string*/
  Ss=3,

  /** long */
  l=4,
};

class Data{
private:
  string str;

public:   
  Data(string str){
    this->str = str;
  }
    

  int getNumber() {
    if("i" == str){
      return 4;
    }
    if ("l" == str){
      return 8;
    }
    if ("f" == str){
      return 4;
    }
    if("Ss" == str){
      return 8; 
    }
  }


};