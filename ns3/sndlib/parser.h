#include <iostream>
#include <string>
#include <string.h>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>

class Parser{
private:
		std :: map <std :: string , std :: tuple <std :: string, std :: string , std :: string, std :: string>> links;
		std :: map <std :: string , std :: tuple<std :: string, std :: string>> nodes;
public:
	std :: map <std :: string , std :: tuple <std :: string, std :: string , std :: string, std :: string>> get_links();
	std :: map <std :: string , std :: tuple<std :: string, std :: string>> get_nodes();
	void read_file();
	std :: vector <std :: string> tokenize(std::string);	
};